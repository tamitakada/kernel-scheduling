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

--context-tokens N makes every prompt long, so the LLM's prefill and decode attention are heavy and compete with Whisper's kernels (with just
the transcript a prompt is a few dozen tokens). N filler tokens of random common words, a pure function of (seed, request index): unique per request
(vLLM's prefix cache cannot reuse them) and identical across runs with the same --seed. The words-per-token ratio is measured with the server's
/tokenize before the run (1.3 is assumed, with a warning, if it is not there). The prompt is N + about 60-100 tokens (framing, template, transcript);
the real count of each request is in the CSV as prompt_tokens (the server's usage report). 0, the default, sends the transcript alone as before.

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
import signal
import sys
import time

import aiohttp

HIST_NAMES = ["vllm:time_to_first_token_seconds", "vllm:request_time_per_output_token_seconds",
              "vllm:inter_token_latency_seconds"]

# --context-tokens: filler made of common English words (almost all one token each), drawn at random.
VOCAB = (
    "action activity age air airport anchor apple area arm art attention autumn baby back "
    "bakery balcony banquet basket berry blanket body book bottle boy bread breeze bridge brother "
    "building business butter button cabbage cabin candle canyon captain car care carnival carrot case "
    "castle center ceremony chance change chapel cheese cherry child choice chorus city class clinic "
    "college community company compass computer condition control corridor cost cottage country couple course court "
    "courtyard culture data day death decision desert development diary difference director doctor door drug "
    "drum east echo education effect effort end energy engine essay event example experience eye "
    "face fact factory family farmhouse father festival field figure film fire flavor flute foot "
    "force forest form fountain friend gallery game garden garlic girl glacier glove government grape "
    "ground group guitar guy hair hammer hand harbor harmony harvest head health heart history "
    "home honey horizon hospital hour house idea image industry information interest island issue jacket "
    "job journey kid kind kitchen ladder lagoon land lantern law leader legend lemon letter "
    "level library life light lighthouse line lot love man market matter meadow melody melon "
    "member message midnight mind minute mirror model moment monastery money month morning mother mountain "
    "movie museum mushroom music name nation need needle news night north novel number ocean "
    "office official oil onion opportunity orange orchard organization others palace paper parade parent part "
    "party passage pasture patient peach peninsula people pepper period person pharmacy phone piano picnic "
    "picture piece pillow place plan plant platform player pocket poem point police policy position "
    "potato power practice president price problem product program project pumpkin quarry question rainbow rate "
    "reason recipe record relationship report research result rhythm ribbon right river road role room "
    "rumor sailor school science season sense service shadow shoe show side signal silence site "
    "situation society son source space spice spring stadium staircase star state station step stone "
    "storm story street student study suitcase summer sunrise sunset support sweater symphony system table "
    "tax teacher team technology temple term test theater thing thunder ticket time tomato tower "
    "town tradition tree trumpet tunnel twilight type umbrella use valley value view village violin "
    "voice volcano voyage wall war warehouse water way week whisper wife windmill window winter "
    "woman word work worker workshop world year"
).split()

FALLBACK_TOKENS_PER_WORD = 1.3                                     # used only if the server has no /tokenize


def context_filler(words, idx, seed):
    """`words` random words, a pure function of (seed, idx): unique per request, the same in every run with the same seed."""
    rng = random.Random(f"context:{seed}:{idx}")
    return " ".join(rng.choices(VOCAB, k=words))


def with_context(text, filler):
    return f"Background text (not relevant, may be ignored):\n{filler}\n\nUtterance:\n{text}"


async def tokenized_count(session, base, model, content):
    """The number of prompt tokens the server's /tokenize reports for this one user message (chat template included), or None."""
    for body in ({"model": model, "messages": [{"role": "user", "content": content}], "add_generation_prompt": True},
                 {"model": model, "prompt": content}):
        try:
            async with session.post(base + "/tokenize", json=body) as resp:
                if resp.status == 200:
                    j = await resp.json(content_type=None)
                    if isinstance(j.get("count"), int):
                        return j["count"]
        except (aiohttp.ClientError, OSError, ValueError):
            pass
    return None


async def calibrate_context(session, base, model, target_tokens, seed):
    """(words per request, tokens per word, how it was found). Two filler sizes are tokenized and the slope taken, so the framing text and the
    chat template (a fixed number of tokens) cancel out."""
    w1, w2 = 100, 400
    c1 = await tokenized_count(session, base, model, with_context("good morning", context_filler(w1, -1, seed)))
    c2 = await tokenized_count(session, base, model, with_context("good morning", context_filler(w2, -1, seed)))
    if c1 is None or c2 is None or c2 <= c1:
        slope, how = FALLBACK_TOKENS_PER_WORD, "ASSUMED (the server has no usable /tokenize): check prompt_tokens in the CSV"
    else:
        slope, how = (c2 - c1) / (w2 - w1), "measured with the server's /tokenize"
    return max(1, round(target_tokens / slope)), slope, how


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
    r = {"idx": idx, "planned_s": offset, "error": None, "n_tokens": 0, "prompt_tokens": None}
    t_send = time.perf_counter()
    r["sent_s"] = t_send - t0
    first = last = None
    chunks = 0
    usage_tokens = None
    usage_prompt = None
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
                    usage_prompt = obj["usage"].get("prompt_tokens")
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
    r["prompt_tokens"] = usage_prompt
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

        ctx_seed = args.seed if args.context_seed is None else args.context_seed
        ctx_words = 0                                                  # --context-tokens: filler words added to every prompt (0: none, the prompt is the transcript)
        if args.context_tokens > 0:
            ctx_words, ctx_slope, ctx_how = await calibrate_context(session, base, model, args.context_tokens, ctx_seed)
            print(f"context: {args.context_tokens} filler tokens requested -> {ctx_words} words per request at {ctx_slope:.2f} tokens/word ({ctx_how}); "
                  f"unique per request, seed {ctx_seed}")

        def body_for(text, i):
            content = with_context(text, context_filler(ctx_words, i, ctx_seed)) if ctx_words else text
            b = {"model": model, "messages": [{"role": "user", "content": content}],
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

        live = {}                                                              # idx -> that request's row, kept up to date as it progresses, so that an
                                                                               # interrupted run still has it

        async def audio_request(i, off):
            """One arrival = one utterance: wait for its planned time, transcribe it, then send the transcript to the LLM."""
            await asyncio.sleep(max(0.0, t0 + off - time.perf_counter()))      # open loop: the utterance arrives at the planned time
            t_arr = time.perf_counter()
            k = i % len(clips)
            r = {"idx": i, "planned_s": off, "sent_s": t_arr - t0, "error": "interrupted", "n_tokens": 0, "ttft_s": None,
                 "tpot_s": None, "e2e_s": 0.0, "llm_sent_s": None, "ttft_e2e_s": None, "total_s": None, "audio_id": ids[k],
                 "asr_s": None, "transcript": "", "transcript_chars": 0, "t_arr": t_arr, "prompt_tokens": None}
            live[i] = r                                                        # "interrupted" stays unless the request runs to its end
            text, err = await transcribe(session, asr_base, asr_model, clips[k], i)
            r.update(asr_s=time.perf_counter() - t_arr, transcript=text or "", transcript_chars=len(text or ""))
            if err or not text:
                r["error"] = "asr: " + (err or "empty transcript")
                r["total_s"] = time.perf_counter() - t_arr
                return r
            r["llm_sent_s"] = time.perf_counter() - t0                         # the transcript goes to the LLM now
            q = await one_request(session, endpoint, body_for(text, i), t0, r["llm_sent_s"], i)                  # sent at once
            r.update(llm_sent_s=q["sent_s"], error=q["error"], n_tokens=q["n_tokens"], ttft_s=q["ttft_s"],
                     tpot_s=q["tpot_s"], e2e_s=q["e2e_s"], prompt_tokens=q["prompt_tokens"])
            r["total_s"] = time.perf_counter() - t_arr
            if q["ttft_s"] is not None:
                r["ttft_e2e_s"] = (t0 + q["sent_s"] - t_arr) + q["ttft_s"]      # the Whisper round trip, then the LLM's TTFT
            return r

        tasks = [asyncio.ensure_future(audio_request(i, off)) for i, off in enumerate(offsets)]
        all_done = asyncio.gather(*tasks)
        interrupted = False
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        try:
            loop.add_signal_handler(signal.SIGINT, stop.set)                   # Ctrl+C stops the experiment but keeps what was measured
        except (NotImplementedError, RuntimeError):
            stop = None                                                        # no signal handlers on this platform: Ctrl+C behaves as it used to
        if stop is None:
            results = await all_done
            wall = time.perf_counter() - t0
        else:
            stopper = asyncio.ensure_future(stop.wait())
            await asyncio.wait({all_done, stopper}, return_when=asyncio.FIRST_COMPLETED)
            interrupted = not all_done.done()
            t_cut = time.perf_counter()
            loop.remove_signal_handler(signal.SIGINT)                          # a second Ctrl+C kills the client at once, as usual
            stopper.cancel()
            if interrupted:
                for t in tasks:
                    t.cancel()                                                 # closes their connections: the servers abort those requests
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.gather(all_done, return_exceptions=True)
                for r in live.values():                                        # what the requests in flight had got to when it stopped
                    if r["error"] == "interrupted":
                        if r["asr_s"] is None:
                            r["asr_s"] = t_cut - r["t_arr"]                    # still at Whisper
                        if r["llm_sent_s"] is not None:
                            r["e2e_s"] = t_cut - (t0 + r["llm_sent_s"])        # at the LLM
                        r["total_s"] = t_cut - r["t_arr"]
                results = [live[i] for i in sorted(live) if live[i]["error"] != "interrupted"]       # the requests that finished
                wall = t_cut - t0
            else:
                results = all_done.result()
                wall = time.perf_counter() - t0
        all_rows = results if not interrupted else [live[i] for i in sorted(live)]     # the CSV: also the ones that were in flight
        after = await scrape(session, base) if args.scrape_metrics else None

    s = summarize(results, wall)
    rate_txt = "all at once" if (args.rate <= 0 or math.isinf(args.rate)) else f"{args.rate:g} req/s ({args.arrival})"
    print(f"\n=== client-side, {s['ok']}/{s['requests']} requests ok, arrivals: {rate_txt}, wall {s['wall_s']:.2f}s ===")
    print(f"TTFT  mean {s['ttft_mean']*1000:9.1f} ms   p50 {s['ttft_p50']*1000:9.1f}   p90 {s['ttft_p90']*1000:9.1f}   p99 {s['ttft_p99']*1000:9.1f}")
    print(f"TPOT  mean {s['tpot_mean']*1000:9.2f} ms   p50 {s['tpot_p50']*1000:9.2f}   p90 {s['tpot_p90']*1000:9.2f}   p99 {s['tpot_p99']*1000:9.2f}"
          f"   (token-weighted {s['tpot_token_weighted']*1000:.2f})")
    print(f"E2E   mean {s['e2e_mean']*1000:9.1f} ms   p99 {s['e2e_p99']*1000:9.1f}   |  {s['output_tokens']} output tokens, {s['throughput_tok_s']:.1f} tok/s")
    pt = [r["prompt_tokens"] for r in results if r.get("prompt_tokens")]
    if pt:
        print(f"PROMPT tokens per request: mean {sum(pt) / len(pt):.0f}, min {min(pt)}, max {max(pt)}"
              + (f"   (--context-tokens {args.context_tokens}: {ctx_words} filler words each)" if ctx_words else ""))
    if s["errors"]:
        print(f"!! {s['errors']} request(s) failed, e.g. {next(r['error'] for r in results if r['error'])}")
    print_asr_summary(results, asr_model)
    if interrupted:
        print(f"\n=== INTERRUPTED by Ctrl+C after {wall:.1f} s: {len(results)} requests had finished (the statistics above); {len(all_rows) - len(results)} were still in "
              f"flight and are in the CSV with error=interrupted (times up to the interrupt); {args.num_requests - len(all_rows)} had not arrived yet and are not in it ===")
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
                        "llm_sent_s", "audio_id", "asr_s", "ttft_e2e_s", "total_s", "transcript_chars", "prompt_tokens"])
            f5 = lambda v: "" if v is None else f"{v:.5f}"
            for r in all_rows:
                w.writerow([r["idx"], f"{r['planned_s']:.4f}", f"{r['sent_s']:.4f}", f5(r["ttft_s"]),
                            "" if r["tpot_s"] is None else f"{r['tpot_s']:.6f}", f"{r['e2e_s']:.5f}", r["n_tokens"], r["error"] or "",
                            f5(r["llm_sent_s"]), r["audio_id"], f5(r["asr_s"]), f5(r["ttft_e2e_s"]), f5(r["total_s"]), r["transcript_chars"],
                            "" if r.get("prompt_tokens") is None else r["prompt_tokens"]])
        print(f"per-request rows written to {args.out}")
    if args.transcripts_out:
        with open(args.transcripts_out, "w") as f:
            for r in all_rows:
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
    ap.add_argument("--context-tokens", type=int, default=0, help="add about this many tokens of unique random filler before the transcript, so "
                    "every prompt is long (heavy prefill and attention); 0: the transcript alone, as before")
    ap.add_argument("--context-seed", type=int, default=None, help="seed of the filler (default: --seed): the same seed gives the same prompts in every run")
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