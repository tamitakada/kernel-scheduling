#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
import math
import random
import statistics
import sys
import time
import pandas as pd

import aiohttp

HIST_NAMES = ["vllm:time_to_first_token_seconds", "vllm:request_time_per_output_token_seconds",
              "vllm:inter_token_latency_seconds"]


def arrival_offsets(n, rate, kind, seed):
    """Seconds after the start at which each request is sent. rate <= 0 or inf: all at once."""
    if rate is None or rate <= 0 or math.isinf(rate):
        return [0.0] * n
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(rate) if kind == "poisson" else 1.0 / rate
    return out


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


async def one_request(session, url, body, t0, offset, idx):
    await asyncio.sleep(max(0.0, t0 + offset - time.perf_counter()))      # open loop: send at the planned time
    r = {"idx": idx, "planned_s": offset, "error": None, "n_tokens": 0}
    t_send = time.perf_counter()
    r["sent_s"] = t_send - t0
    first = last = None
    chunks = 0
    usage_tokens = None
    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
            
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                now = time.perf_counter()
                if obj.get("usage"):
                    usage_tokens = obj["usage"].get("completion_tokens")
                for ch in obj.get("choices", []):
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                        if first is None:
                            first = now
                        last = now
                        chunks += 1
    except Exception as e:
        r["error"] = repr(e)
    t_end = time.perf_counter()
    n = usage_tokens if usage_tokens is not None else chunks
    r["n_tokens"] = n
    r["e2e_s"] = t_end - t_send
    r["ttft_s"] = (first - t_send) if first is not None else None
    r["tpot_s"] = ((last - first) / (n - 1)) if (first is not None and n and n > 1) else None
    return r


def parse_hist(text, names=HIST_NAMES):
    out = {n: [0.0, 0.0] for n in names}                   # name -> [sum, count], summed over all label sets
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        for n in names:
            if line.startswith(n + "_sum"):
                out[n][0] += float(line.rsplit(" ", 1)[1])
            elif line.startswith(n + "_count"):
                out[n][1] += float(line.rsplit(" ", 1)[1])
    return out


async def scrape(session, base):
    try:
        async with session.get(base + "/metrics") as resp:
            if resp.status != 200:
                return None
            return parse_hist(await resp.text())
    except Exception:
        return None


async def resolve_model(session, base, model):
    if model:
        return model
    async with session.get(base + "/v1/models") as resp:
        return (await resp.json())["data"][0]["id"]


def summarize(results, wall_s):
    ok = [r for r in results if r["error"] is None and r["ttft_s"] is not None]
    ttft = sorted(r["ttft_s"] for r in ok)
    tpot_rs = [r for r in ok if r["tpot_s"] is not None]
    tpot = sorted(r["tpot_s"] for r in tpot_rs)
    e2e = sorted(r["e2e_s"] for r in ok)
    dec_tokens = sum(r["n_tokens"] - 1 for r in tpot_rs)
    dec_time = sum(r["tpot_s"] * (r["n_tokens"] - 1) for r in tpot_rs)
    out_tokens = sum(r["n_tokens"] for r in ok)
    mean = lambda v: (sum(v) / len(v)) if v else float("nan")
    return {
        "requests": len(results), "ok": len(ok), "errors": len(results) - len(ok),
        "ttft_mean": mean(ttft), "ttft_p50": percentile(ttft, 50), "ttft_p90": percentile(ttft, 90), "ttft_p99": percentile(ttft, 99),
        "tpot_mean": mean(tpot), "tpot_p50": percentile(tpot, 50), "tpot_p90": percentile(tpot, 90), "tpot_p99": percentile(tpot, 99),
        "tpot_token_weighted": (dec_time / dec_tokens) if dec_tokens else float("nan"),
        "e2e_mean": mean(e2e), "e2e_p99": percentile(e2e, 99),
        "output_tokens": out_tokens, "wall_s": wall_s,
        "throughput_tok_s": out_tokens / wall_s if wall_s > 0 else float("nan"),
    }


async def run(args):
    dataset_df = pd.read_csv("hf://datasets/fka/prompts.chat/prompts.csv")
    prompts = dataset_df["prompt"].sample(n=100, random_state=42).tolist()

    base = args.url.rstrip("/")
    endpoint = base + "/v1/chat/completions"

    connector = aiohttp.TCPConnector(limit=0)                          # no cap on concurrent connections: open loop
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        model = await resolve_model(session, base, args.model)
        before = await scrape(session, base) if args.scrape_metrics else None
        offsets = arrival_offsets(len(prompts), args.rate, args.arrival, 42)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            one_request(session, 
                        endpoint, 
                        {"model": model,
                         "messages": [{"role": "user", "content": prompts[i]}],
                         "max_tokens": 2048,
                         "temperature": 0.0,
                         "seed": 42,
                         "chat_template_kwargs": {"enable_thinking": False},
                         "ignore_eos": True,
                         "stream": True,
                         "stream_options": {"include_usage": True}}, 
                        t0, off, i) 
            for i, off in enumerate(offsets)])
        wall = time.perf_counter() - t0
        after = await scrape(session, base) if args.scrape_metrics else None

    s = summarize(results, wall)
    rate_txt = "all at once" if (args.rate <= 0 or math.isinf(args.rate)) else f"{args.rate:g} req/s ({args.arrival})"
    print(f"\n=== client-side, {s['ok']}/{s['requests']} requests ok, arrivals: {rate_txt}, wall {s['wall_s']:.2f}s ===")
    print(f"TTFT  mean {s['ttft_mean']*1000:9.1f} ms   p50 {s['ttft_p50']*1000:9.1f}   p90 {s['ttft_p90']*1000:9.1f}   p99 {s['ttft_p99']*1000:9.1f}")
    print(f"TPOT  mean {s['tpot_mean']*1000:9.2f} ms   p50 {s['tpot_p50']*1000:9.2f}   p90 {s['tpot_p90']*1000:9.2f}   p99 {s['tpot_p99']*1000:9.2f}"
          f"   (token-weighted {s['tpot_token_weighted']*1000:.2f})")
    print(f"E2E   mean {s['e2e_mean']*1000:9.1f} ms   p99 {s['e2e_p99']*1000:9.1f}   |  {s['output_tokens']} output tokens, {s['throughput_tok_s']:.1f} tok/s")
    if s["errors"]:
        print(f"!! {s['errors']} request(s) failed, e.g. {next(r['error'] for r in results if r['error'])}")
    if before and after:
        print("--- server-side (vLLM's own histograms, delta over the measured run) ---")
        for n, label in (("vllm:time_to_first_token_seconds", "TTFT"), ("vllm:request_time_per_output_token_seconds", "TPOT (per-request mean)"),
                         ("vllm:inter_token_latency_seconds", "inter-token latency")):
            ds, dc = after[n][0] - before[n][0], after[n][1] - before[n][1]
            print(f"{label:<26} mean {ds / dc * 1000:9.2f} ms over {int(dc)} observations" if dc > 0 else f"{label:<26} (no observations)")
    elif args.scrape_metrics:
        print("(server /metrics not available; server-side averages skipped)")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "planned_s", "sent_s", "ttft_s", "tpot_s", "e2e_s", "n_tokens", "error"])
            for r in results:
                w.writerow([r["idx"], f"{r['planned_s']:.4f}", f"{r['sent_s']:.4f}",
                            "" if r["ttft_s"] is None else f"{r['ttft_s']:.5f}", "" if r["tpot_s"] is None else f"{r['tpot_s']:.6f}",
                            f"{r['e2e_s']:.5f}", r["n_tokens"], r["error"] or ""])
        print(f"per-request rows written to {args.out}")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default=None, help="default: the first model the server lists")
    ap.add_argument("--rate", type=float, default=2.0, help="requests per second; 0 or inf = all at once")
    ap.add_argument("--arrival", choices=["poisson", "fixed"], default="poisson")
    ap.add_argument("--no-scrape-metrics", dest="scrape_metrics", action="store_false")
    ap.add_argument("--out", default=None, help="CSV of per-request results")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
