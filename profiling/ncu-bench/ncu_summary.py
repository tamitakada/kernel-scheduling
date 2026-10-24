#!/usr/bin/env python3
"""
Summarize one or more Nsight Compute reports (.ncu-rep).

Launches are grouped by unique (kernel name, block size) pair, pooled across every grid size
and every input file. For each group it reports the mean and sample standard deviation of
duration-per-wave and of each metric (std is 0 when a group has a single launch).

Duration/Wave is computed per launch as gpu__time_duration (in microseconds) divided by
ceil(waves), where waves = launch__waves_per_multiprocessor. If that metric is missing, waves is
computed as grid_volume / (SM_count * blocks_per_SM) from the occupancy-limit metrics (SM count can
be overridden with --sm-count). Launches for which waves can't be determined are excluded from this
column only. The column is then averaged over all launches in the group.

Blocks/Wave is computed per launch as grid size (launch__grid_size, or x*y*z of the grid dims if
that metric is absent) divided by the same waves value; i.e. the number of blocks that fit in one wave
(SM count x resident blocks per SM). It is averaged over launches like the other columns.

Because the mean is taken over all pooled launches, it equals the launch-count-weighted
average of the per-(name, grid, block) means. The std is computed over the same pooled
launches, so it includes variation between different grid sizes.

Usage:
    python ncu_summary.py report.ncu-rep
    python ncu_summary.py run1.ncu-rep run2.ncu-rep run3.ncu-rep --csv out.csv
    python ncu_summary.py runs/*.ncu-rep --csv out.csv
    python ncu_summary.py report.ncu-rep --ncu-path "/opt/nvidia/nsight-compute/2024.3.0/extras/python"

Uses the `ncu_report` Python module that ships with Nsight Compute
(<install dir>/extras/python). No pip packages needed.
"""
import argparse
import csv
import glob
import os
import math
import sys
from collections import defaultdict
from statistics import mean, stdev

# Column label -> candidate NCU metric names (the first one present in the report is used).
METRICS = {
    "SM Util %":        ("sm__throughput.avg.pct_of_peak_sustained_elapsed",),
    "Achieved Occ %":   ("sm__warps_active.avg.pct_of_peak_sustained_active",),
    "IPC (active)":     ("sm__inst_executed.avg.per_cycle_active",),
    "L2 Throughput %":  ("lts__throughput.avg.pct_of_peak_sustained_elapsed",),
    "L2 Hit Rate %":    ("lts__t_sector_hit_rate.pct",),
    "DRAM BW Util %":   ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                         "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
}

DURATION_METRIC = "gpu__time_duration.sum"
RATIO_LABEL = "Duration/Wave (us)"
BPW_LABEL = "Blocks/Wave"
WAVES_METRIC = "launch__waves_per_multiprocessor"
SM_COUNT_METRIC = "device__attribute_multiprocessor_count"
OCC_LIMIT_METRICS = ("launch__occupancy_limit_blocks", "launch__occupancy_limit_registers",
                     "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps")

# Time metrics are converted to microseconds using the unit the report states.
TIME_UNIT_TO_US = {
    "nsecond": 1e-3, "ns": 1e-3,
    "usecond": 1.0,  "us": 1.0,
    "msecond": 1e3,  "ms": 1e3,
    "second": 1e6,   "s": 1e6,
}

SEARCH_GLOBS = [
    "/opt/nvidia/nsight-compute/*/extras/python",
    "/usr/local/cuda*/nsight-compute*/extras/python",
    "/usr/local/NVIDIA-Nsight-Compute*/extras/python",
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute *\extras\python",
]


def import_ncu_report(extra_path=None):
    paths = []
    if extra_path:
        paths.append(extra_path)
    if os.environ.get("NCU_PYTHON_PATH"):
        paths.append(os.environ["NCU_PYTHON_PATH"])
    try:
        import ncu_report  # already on sys.path?
        return ncu_report
    except ImportError:
        pass
    for g in SEARCH_GLOBS:
        paths.extend(sorted(glob.glob(g), reverse=True))  # newest first
    for p in paths:
        if os.path.isdir(p):
            sys.path.insert(0, p)
            try:
                import ncu_report
                return ncu_report
            except ImportError:
                sys.path.pop(0)
    sys.exit(
        "Could not import ncu_report. Point to it with --ncu-path or NCU_PYTHON_PATH, "
        "e.g. /opt/nvidia/nsight-compute/<version>/extras/python"
    )


def get_value(action, metric_name):
    m = action.metric_by_name(metric_name)
    if m is None:
        return None
    try:
        v = m.as_double()
    except Exception:
        try:
            v = float(m.as_uint64())
        except Exception:
            return None
    try:
        unit = str(m.unit()).strip().lower()
    except Exception:
        unit = ""
    factor = TIME_UNIT_TO_US.get(unit)
    if factor is None and metric_name.startswith("gpu__time_duration"):
        factor = 1e-3  # ncu's raw unit for durations is nanoseconds
    return v * factor if factor is not None else v


def get_dims(action, prefix):
    """Return (x, y, z) for 'grid' or 'block'; falls back to the total size if dims are unavailable."""
    dims = [get_value(action, f"launch__{prefix}_dim_{a}") for a in "xyz"]
    if all(d is not None for d in dims):
        return tuple(int(d) for d in dims)
    total = get_value(action, f"launch__{prefix}_size")
    return (int(total),) if total is not None else None


def get_waves(action, grid, sm_count_override=None):
    """Waves per SM for a launch: from the report if usable, else computed from occupancy limits.
    Non-finite or non-positive values (e.g. inf when an occupancy limit is 0) are treated as missing."""
    w = get_value(action, WAVES_METRIC)
    if w is not None and math.isfinite(w) and w > 0:
        return w
    sm_count = sm_count_override or get_value(action, SM_COUNT_METRIC)
    limits = [v for v in (get_value(action, m) for m in OCC_LIMIT_METRICS)
              if v is not None and math.isfinite(v) and v > 0]
    if grid and sm_count and math.isfinite(sm_count) and limits:
        w = math.prod(grid) / (sm_count * min(limits))
        if math.isfinite(w) and w > 0:
            return w
    return None


def fmt_dims(d):
    if d is None:
        return "n/a"
    if len(d) == 3:
        return f"({d[0]}, {d[1]}, {d[2]})"
    return f"{d[0]} total"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+", help="One or more .ncu-rep files (launches are pooled)")
    ap.add_argument("--ncu-path", help="Path to Nsight Compute's extras/python directory")
    ap.add_argument("--csv", help="Also write results to this CSV file")
    ap.add_argument("--sm-count", type=int,
                    help="SM count to use if waves must be computed (default: read from the report)")
    ap.add_argument("--find", nargs="+", metavar="SUBSTR",
                    help="List metric names in the (first) report containing any of these substrings "
                         "(e.g. --find dram shared) and exit")
    args = ap.parse_args()

    ncu_report = import_ncu_report(args.ncu_path)

    if args.find:
        ctx = ncu_report.load_report(args.reports[0])
        names = set()
        for i in range(ctx.num_ranges()):
            rng = ctx.range_by_idx(i)
            if rng.num_actions():
                names.update(rng.action_by_idx(0).metric_names())
                break
        pats = [x.lower() for x in args.find]
        hits = sorted(n for n in names if any(p in n.lower() for p in pats))
        print("\n".join(hits) if hits else "No matching metrics in report.")
        return

    # (kernel name, block dims) -> metric label -> list of per-launch values, pooled over all grids/files
    data = defaultdict(lambda: defaultdict(list))
    launches = defaultdict(int)
    files_loaded = 0
    no_waves = 0

    for path in args.reports:
        try:
            ctx = ncu_report.load_report(path)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}", file=sys.stderr)
            continue
        files_loaded += 1
        n_before = sum(launches.values())
        for i in range(ctx.num_ranges()):
            rng = ctx.range_by_idx(i)
            for j in range(rng.num_actions()):
                action = rng.action_by_idx(j)
                key = (action.name(), get_dims(action, "block"))
                launches[key] += 1
                dur_us = get_value(action, DURATION_METRIC)
                grid = get_dims(action, "grid")
                waves = get_waves(action, grid, args.sm_count)
                grid_total = get_value(action, "launch__grid_size")
                if grid_total is None and grid:
                    grid_total = math.prod(grid)
                if waves is not None and grid_total is not None and math.isfinite(grid_total):
                    data[key][BPW_LABEL].append(grid_total / waves)
                if dur_us is not None and math.isfinite(dur_us) and waves is not None:
                    data[key][RATIO_LABEL].append(dur_us / max(1, math.ceil(waves - 1e-6)))
                else:
                    no_waves += 1
                for label, candidates in METRICS.items():
                    for metric in candidates:
                        v = get_value(action, metric)
                        if v is not None:
                            data[key][label].append(v)
                            break
        print(f"Loaded {path}: {sum(launches.values()) - n_before} kernel launches", file=sys.stderr)

    if files_loaded == 0:
        sys.exit("No reports could be loaded.")
    print(f"Pooled {sum(launches.values())} launches from {files_loaded} file(s), "
          f"{len(launches)} unique kernel/block groups.", file=sys.stderr)

    if not launches:
        sys.exit("No kernels found in report(s).")
    if no_waves:
        print(f"Note: {no_waves} launch(es) lacked duration or wave info and are excluded from "
              f"'{RATIO_LABEL}'. (Check with: --find waves occupancy_limit multiprocessor)",
              file=sys.stderr)

    labels = [RATIO_LABEL, BPW_LABEL] + list(METRICS)
    rows = []
    for key in sorted(launches, key=lambda k: (k[0], k[1] or ())):
        name, block = key
        row = {"Kernel": name, "Block": fmt_dims(block), "Launches": launches[key]}
        for label in labels:
            vals = data[key].get(label)
            row[label] = mean(vals) if vals else None
            row[f"{label} std"] = (stdev(vals) if len(vals) > 1 else 0.0) if vals else None
        rows.append(row)

    def fmt(v, label=None):
        if v is None:
            return "n/a"
        return f"{v:.4g}" if label == RATIO_LABEL else f"{v:.2f}"  # ratio is tiny; keep significant digits

    for r in rows:
        print(f"\n{r['Kernel']}\n  block {r['Block']}  (launches: {r['Launches']})")
        for label in labels:
            print(f"  {label:<30} {fmt(r[label], label)}  \u00b1 {fmt(r[label + ' std'], label)}")

    missing = [l for l in labels if all(r[l] is None for r in rows)]
    if missing:
        print("\nNote: no data for: " + ", ".join(missing) +
              "\n(find what your report has with: python ncu_summary.py report.ncu-rep --find dram shared)")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Kernel", "Block", "Launches"]
                               + [c for l in labels for c in (l, f"{l} std")])
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()