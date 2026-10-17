#!/usr/bin/env python3
"""
Open-loop load generator for two running `vllm serve` processes sharing one GPU, driven by SPEECH: the first n utterances of
LibriSpeech test-clean (openslr/librispeech_asr, streamed) arrive on a schedule (Poisson or fixed rate), NOT all at once. Each one is
sent to the Whisper server (serve_whisper.sh, /v1/audio/transcriptions) and the transcript goes to the LLM server
(serve_online.sh, /v1/chat/completions) as the request, streamed so that time-to-first-token and per-token timing are observed
the way a user sees them.

  python online_client.py --num-requests 50 --rate 2 --max-tokens 128 --temperature 0 --ignore-eos --no-thinking \\
         --out results.csv

Start scheduler_daemon, the LLM server and the Whisper server first (see serve_whisper.sh for the order and the memory
split). The client is plain HTTP: it needs `datasets` to stream LibriSpeech and nothing else; nothing is written to disk. There is no warm-up: the first
requests include first-use costs.

Reports, per request and averaged:
  TTFT  time from sending the transcript to the first content chunk (client side: includes HTTP, queueing, prefill)
  TPOT  (time of last token - time of first token) / (output tokens - 1), the usual definition; the first token belongs
        to TTFT. Output tokens come from the server's usage report. This assumes one token per streamed chunk (vLLM's
        default); if chunks carry k tokens, TPOT reads low by up to about (k-1)/(n-1) of itself.
  ASR   the Whisper request's round trip (it includes the Whisper server's own queueing), and the end-to-end TTFT from the
        utterance's arrival to the first LLM token.
Averages are per-request means; the token-weighted TPOT (total decode time / total decode tokens) is shown as well.

It also scrapes the LLM server's Prometheus histograms before and after the measured run and prints its server-side mean
TTFT / TPOT (delta of _sum over delta of _count), which is what vLLM itself recorded for these requests.
Everything the scheduler adds (kernel queueing, ...) is inside both views, since both are wall-clock.
"""
import argparse
import asyncio
import csv
import json
import math
import random
import sys
import time

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


def sample_librispeech(n):
    """The first n LibriSpeech test-clean utterances as (ids, raw FLAC bytes). Streamed: nothing is downloaded in full or
    cached on disk. decode=False hands over each utterance's FLAC file as stored (and keeps `datasets` from decoding it)."""
    from datasets import Audio, load_dataset
    rows = list(load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
                .cast_column("audio", Audio(decode=False)).take(n))
    return [r["id"] for r in rows], [r["audio"]["bytes"] for r in rows]


async def transcribe(session, base, model, flac, i):
    """One request to the Whisper server. Returns (text, None) or (None, error)."""
    form = aiohttp.FormData()
    form.add_field("file", flac, filename=f"{i}.flac", content_type="audio/flac")
    for k, v in (("model", model), ("language", "en"), ("response_format", "json"), ("temperature", "0")):
        form.add_field(k, v)
    try:
        async with session.post(base + "/v1/audio/transcriptions", data=form) as resp:
            if resp.status != 200:
                return None, f"HTTP {resp.status}: {(await resp.text())[:200]}"
            return ((await resp.json(content_type=None)).get("text") or "").strip(), None
    except Exception as e:
        return None, repr(e)


def _stat_line(label, vals, note):
    vals = sorted(vals)
    if not vals:
        return f"{label:<10} (no observations)"
    return (f"{label:<10} mean {sum(vals) / len(vals) * 1000:9.1f} ms   p50 {percentile(vals, 50) * 1000:9.1f}   "
            f"p90 {percentile(vals, 90) * 1000:9.1f}   p99 {percentile(vals, 99) * 1000:9.1f}   {note}")


def print_asr_summary(results, asr_model):
    failed = [r for r in results if (r["error"] or "").startswith("asr:")]
    reached = [r for r in results if r["ttft_s"] is not None]
    print(f"\n=== audio -> Whisper ({asr_model}) -> LLM: {len(results) - len(failed)}/{len(results)} transcribed, {len(reached)} answered ===")
    print(_stat_line("ASR", [r["asr_s"] for r in results if r not in failed], "(round trip to the Whisper server, incl. its queueing)"))
    print(_stat_line("E2E TTFT", [r["ttft_e2e_s"] for r in reached], "(audio arrival -> first LLM token)"))
    if failed:
        print(f"!! {len(failed)} request(s) failed in the ASR stage, e.g. {failed[0]['error']}")


async def run(args):
    ids, clips = sample_librispeech(args.num_requests)                    # before anything is measured: request i is utterance i (the split has
                                                                       # 2620; a longer run goes round again)
    base, asr_base = args.url.rstrip("/"), args.asr_url.rstrip("/")
    endpoint = base + "/v1/chat/completions"
    connector = aiohttp.TCPConnector(limit=0)                          # no cap on concurrent connections: open loop
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        model = await resolve_model(session, base, args.model)
        try:                                                           # fail early, and say how to fix it, if Whisper is not up
            asr_model = args.asr_model or await resolve_model(session, asr_base, None)
            async with session.get(asr_base + "/v1/models") as resp:
                resp.raise_for_status()
        except (aiohttp.ClientError, OSError) as e:
            raise SystemExit(f"error: cannot reach the Whisper server at {asr_base} ({e!r}); start serve_whisper.sh")

        def body_for(text):
            b = {"model": model, "messages": [{"role": "user", "content": text}],
                 "max_tokens": args.max_tokens, "temperature": args.temperature, "stream": True,
                 "stream_options": {"include_usage": True}}
            if args.seed is not None:
                b["seed"] = args.seed
            if args.no_thinking:
                b["chat_template_kwargs"] = {"enable_thinking": False}
            if args.ignore_eos:
                b["ignore_eos"] = True
            return b

        before = await scrape(session, base) if args.scrape_metrics else None
        offsets = arrival_offsets(args.num_requests, args.rate, args.arrival, args.seed)
        t0 = time.perf_counter()

        async def audio_request(i, off):
            """One arrival = one utterance: wait for its planned time, transcribe it, then send the transcript to the LLM."""
            await asyncio.sleep(max(0.0, t0 + off - time.perf_counter()))      # open loop: the utterance arrives at the planned time
            t_arr = time.perf_counter()
            k = i % len(clips)
            text, err = await transcribe(session, asr_base, asr_model, clips[k], i)
            r = {"idx": i, "planned_s": off, "sent_s": t_arr - t0, "error": None, "n_tokens": 0, "ttft_s": None,
                 "tpot_s": None, "e2e_s": 0.0, "llm_sent_s": None, "ttft_e2e_s": None, "total_s": None, "audio_id": ids[k],
                 "asr_s": time.perf_counter() - t_arr, "transcript": text or "", "transcript_chars": len(text or "")}
            if err or not text:
                r["error"] = "asr: " + (err or "empty transcript")
                r["total_s"] = time.perf_counter() - t_arr
                return r
            q = await one_request(session, endpoint, body_for(text), t0, time.perf_counter() - t0, i)         # sent at once
            r.update(llm_sent_s=q["sent_s"], error=q["error"], n_tokens=q["n_tokens"], ttft_s=q["ttft_s"],
                     tpot_s=q["tpot_s"], e2e_s=q["e2e_s"])
            r["total_s"] = time.perf_counter() - t_arr
            if q["ttft_s"] is not None:
                r["ttft_e2e_s"] = (t0 + q["sent_s"] - t_arr) + q["ttft_s"]      # the Whisper round trip, then the LLM's TTFT
            return r

        results = await asyncio.gather(*[audio_request(i, off) for i, off in enumerate(offsets)])
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
    print_asr_summary(results, asr_model)
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
            w.writerow(["idx", "planned_s", "sent_s", "ttft_s", "tpot_s", "e2e_s", "n_tokens", "error",
                        "llm_sent_s", "audio_id", "asr_s", "ttft_e2e_s", "total_s", "transcript_chars"])
            f5 = lambda v: "" if v is None else f"{v:.5f}"
            for r in results:
                w.writerow([r["idx"], f"{r['planned_s']:.4f}", f"{r['sent_s']:.4f}", f5(r["ttft_s"]),
                            "" if r["tpot_s"] is None else f"{r['tpot_s']:.6f}", f"{r['e2e_s']:.5f}", r["n_tokens"], r["error"] or "",
                            f5(r["llm_sent_s"]), r["audio_id"], f5(r["asr_s"]), f5(r["ttft_e2e_s"]), f5(r["total_s"]), r["transcript_chars"]])
        print(f"per-request rows written to {args.out}")
    if args.transcripts_out:
        with open(args.transcripts_out, "w") as f:
            for r in results:
                f.write(json.dumps({"idx": r["idx"], "audio_id": r["audio_id"], "transcript": r["transcript"],
                                    "asr_s": r["asr_s"], "error": r["error"]}) + "\n")
        print(f"transcripts written to {args.transcripts_out}")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000", help="the LLM server")
    ap.add_argument("--model", default=None, help="default: the first model the LLM server lists")
    ap.add_argument("--asr-url", default="http://localhost:8001", help="the Whisper server (serve_whisper.sh)")
    ap.add_argument("--asr-model", default=None, help="default: the first model the Whisper server lists")
    ap.add_argument("--num-requests", type=int, default=50, help="how many LibriSpeech utterances to take = how many requests")
    ap.add_argument("--rate", type=float, default=2.0, help="requests per second; 0 or inf = all at once")
    ap.add_argument("--arrival", choices=["poisson", "fixed"], default="poisson")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="seeds the arrival process and the LLM sampling")
    ap.add_argument("--ignore-eos", action="store_true", help="always generate max-tokens (equal lengths across runs)")
    ap.add_argument("--no-thinking", action="store_true", help='send chat_template_kwargs {"enable_thinking": false} (Qwen3)')
    ap.add_argument("--no-scrape-metrics", dest="scrape_metrics", action="store_false")
    ap.add_argument("--out", default=None, help="CSV of per-request results")
    ap.add_argument("--transcripts-out", default=None, help="JSONL with the transcript and Whisper round trip of every request")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()