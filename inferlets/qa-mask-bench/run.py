#!/usr/bin/env python3
"""Driver for the masking experiment: one `pie run` invocation per (round, case).

Each invocation builds that round's base cache once and runs all of the round's
asks in-process (so the cache is reused across asks, which is the whole point).
The (case, round) payload is passed as a single JSON `--input` object; the
inferlet deserializes it into `{case, round, max_tokens?}`.

The inferlet is a `wasm32-wasip2` component built from `src/lib.rs`. This script
builds it once (`cargo build --release --target wasm32-wasip2`, exactly what
`pie build` does for Rust) and reuses the artifact for every invocation. Pass
`--wasm <path>` to skip the build.

Usage from anywhere:
  python inferlets/qa-mask-bench/run.py --cases 1,2,3,4,5,6 --rounds :

The model is whatever `~/.pie/config.toml` targets (this host: Qwen/Qwen3-0.6B
on cuda_native). `--model` is accepted for back-compat but only recorded in the
output rows — `pie run` selects the model from the config, not the CLI.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parents[1]  # inferlets/qa-mask-bench -> inferlets -> <repo root>
DEFAULT_PIE = REPO_ROOT / "target" / "release" / "pie"
MANIFEST = HERE / "Pie.toml"
WASM_ARTIFACT = HERE / "target" / "wasm32-wasip2" / "release" / "qa_mask_bench.wasm"

QA_BLOCK_RE = re.compile(r"\[QA_RESULT\](.*?)\[/QA_RESULT\]", re.DOTALL)
INSTANCE_PREFIX_RE = re.compile(r"^\s*\[Instance [^\]]+\]\s?")


def parse_qa_blocks(stdout: str):
    results = []
    for m in QA_BLOCK_RE.finditer(stdout):
        d = {}
        for line in m.group(1).splitlines():
            line = INSTANCE_PREFIX_RE.sub("", line).strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
        if d:
            results.append(d)
    return results


def parse_range(s: str, total: int):
    if ":" in s:
        a, b = s.split(":", 1)
        a = int(a) if a else 0
        b = int(b) if b else total
        return list(range(a, min(b, total)))
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    return [int(s)]


def build_wasm() -> Path:
    sys.stderr.write("Building inferlet (cargo build --release --target wasm32-wasip2)...\n")
    proc = subprocess.run(
        ["cargo", "build", "--release", "--target", "wasm32-wasip2"],
        cwd=HERE, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        sys.exit("cargo build failed")
    if not WASM_ARTIFACT.exists():
        sys.exit(f"expected wasm not found at {WASM_ARTIFACT}")
    return WASM_ARTIFACT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="(from ~/.pie/config.toml)",
                    help="Recorded in output rows only; pie run picks the model from its config.")
    ap.add_argument("--cases", default="1,2,3,4,5,6")
    ap.add_argument("--rounds", default=":",
                    help="Round index range, e.g. 0:5 or 2 or 0,3. Default: all.")
    ap.add_argument("--rounds-json", default=str(HERE / "rounds.json"))
    ap.add_argument("--output", default=str(HERE / "results.jsonl"))
    ap.add_argument("--pie", default=str(DEFAULT_PIE),
                    help="Path to the pie binary (default: repo target/release/pie).")
    ap.add_argument("--wasm", default=None,
                    help="Prebuilt .wasm component. If omitted, the inferlet is built once.")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--print-cmd", action="store_true",
                    help="Print the pie command(s) and exit.")
    args = ap.parse_args()

    rpath = Path(args.rounds_json)
    if not rpath.exists():
        sys.stderr.write(f"{rpath} not found — run prepare.py first.\n")
        sys.exit(1)
    rounds = json.loads(rpath.read_text())
    if not rounds:
        sys.stderr.write(f"{rpath} is empty — re-run prepare.py.\n")
        sys.exit(1)

    ridxs = parse_range(args.rounds, len(rounds))
    cases = [int(c) for c in args.cases.split(",")]

    wasm = Path(args.wasm) if args.wasm else build_wasm()

    out_path = Path(args.output)
    if not args.append and out_path.exists():
        out_path.unlink()

    n_total = len(ridxs) * len(cases)
    n_done = 0
    t0 = time.time()
    env = os.environ.copy()

    with out_path.open("a") as out_f:
        for ri in ridxs:
            rd = rounds[ri]
            for case in cases:
                n_done += 1
                payload = {"case": case, "round": rd, "max_tokens": args.max_tokens}
                cmd = [
                    args.pie, "run",
                    "--path", str(wasm),
                    "--manifest", args.manifest,
                    "--input", json.dumps(payload, separators=(",", ":")),
                    "--stdout", "--quiet",
                ]
                if args.print_cmd:
                    print(f"# round={ri} case={case}")
                    print(" ".join(shlex.quote(c) for c in cmd))
                    print()
                    continue
                print(f"[{n_done}/{n_total}] round={ri} case={case} ...", flush=True)
                t_start = time.time()
                try:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True,
                        timeout=args.timeout, env=env,
                    )
                except subprocess.TimeoutExpired:
                    sys.stderr.write(f"  TIMEOUT round={ri} case={case}\n")
                    continue
                wall = time.time() - t_start

                blocks = parse_qa_blocks(proc.stdout + "\n" + proc.stderr)
                if not blocks:
                    sys.stderr.write(
                        f"  No [QA_RESULT] block round={ri} case={case} "
                        f"(rc={proc.returncode}); stderr tail:\n"
                    )
                    sys.stderr.write(proc.stderr[-800:] + "\n")
                    continue
                for b in blocks:
                    b["round_idx"] = ri
                    b["model"] = args.model
                    b["wall_ms_round"] = int(wall * 1000)
                    out_f.write(json.dumps(b) + "\n")
                    out_f.flush()
                print(f"   {len(blocks)} asks  ({wall:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nDone. {n_done} invocations in {elapsed:.0f}s. Results → {out_path}")


if __name__ == "__main__":
    main()
