#!/usr/bin/env python3
"""Score results.jsonl for the masking experiment.

Each result row is one *ask* (two questions). We split it into two question-level
records for accuracy (EM/F1, HotpotQA-style normalization) and keep ask-level
timing (prefill TTFT, decode total, per-token decode, register cost).

Reports a per-case table of accuracy + latency, the headline numbers the
experiment is about.
"""
import argparse
import json
import re
import string
import sys
from collections import defaultdict
from collections import Counter
from pathlib import Path


def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    for tok in ("<|im_end|>", "<|endoftext|>", "</s>"):
        s = s.replace(tok, "")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[" + re.escape(string.punctuation) + "]", " ", s)
    s = s.replace(",", " ")  # so "52,310" == "52310"
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def em_score(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def to_num(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    rows = []
    with open(args.results) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        sys.stderr.write("No results to score.\n")
        sys.exit(1)

    # Per-case accumulators.
    qrec = defaultdict(list)   # case -> list of (em, f1) per question
    arec = defaultdict(list)   # case -> list of ask timing dicts

    for r in rows:
        case = r.get("case", "?")
        # Two questions per ask.
        for ai, gi in (("ans1", "gold1"), ("ans2", "gold2")):
            pred, gold = r.get(ai, ""), r.get(gi, "")
            qrec[case].append((em_score(pred, gold), f1_score(pred, gold)))
        arec[case].append({
            k: to_num(r.get(k))
            for k in ("register_ms", "prefill_ms", "decode_ms", "per_token_us",
                      "output_tokens", "attended_tokens", "kv_resident_tokens",
                      "prompt_tokens", "sys_tokens", "info_tokens")
        })

    def avg(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    cases = sorted(qrec.keys(), key=str)
    labels = {
        "1": "1 all-cached",
        "2": "2 mask+drop casc",
        "3": "3 mask+drop excl",
        "4": "4 inline",
        "5": "5 pure-mask casc",
        "6": "6 pure-mask excl",
    }

    print("Accuracy + latency by case "
          "(N = questions; latency averaged per ask):\n")
    print(f"{'Case':<18}{'Nq':>5}{'EM':>8}{'F1':>8}"
          f"{'Reg ms':>9}{'Prefill ms':>12}{'Decode ms':>11}{'Tok µs':>9}"
          f"{'OutTok':>8}{'Attend':>8}{'Resident':>10}")
    print("-" * 114)
    for c in cases:
        qs = qrec[c]
        ts = arec[c]
        em = sum(x[0] for x in qs) / len(qs)
        f1 = sum(x[1] for x in qs) / len(qs)
        print(f"{labels.get(str(c), str(c)):<18}{len(qs):>5}{em:>8.3f}{f1:>8.3f}"
              f"{avg([t['register_ms'] for t in ts]):>9.0f}"
              f"{avg([t['prefill_ms'] for t in ts]):>12.1f}"
              f"{avg([t['decode_ms'] for t in ts]):>11.1f}"
              f"{avg([t['per_token_us'] for t in ts]):>9.0f}"
              f"{avg([t['output_tokens'] for t in ts]):>8.1f}"
              f"{avg([t['attended_tokens'] for t in ts]):>8.0f}"
              f"{avg([t['kv_resident_tokens'] for t in ts]):>10.0f}")

    # --- Token breakdown by part ---
    print("\nToken breakdown by part (per ask):\n")
    print(f"{'Case':<18}{'System':>8}{'Used info':>11}{'Query':>8}{'Output':>8}"
          f"{'Cached base':>13}")
    print("-" * 66)
    for c in cases:
        ts = arec[c]
        sys_t = avg([t["sys_tokens"] for t in ts])
        info_t = avg([t["info_tokens"] for t in ts])
        prompt_t = avg([t["prompt_tokens"] for t in ts])
        out_t = avg([t["output_tokens"] for t in ts])
        # Cached base = what was prefilled once per round and reused.
        # Cases 1/2/3/5/6 cache system+all infos; case 4 caches system only.
        per_info = info_t / 2 if info_t == info_t else float("nan")  # 2 used infos
        n_items = 6
        base = sys_t + (per_info * n_items if str(c) != "4" else 0)
        print(f"{labels.get(str(c), str(c)):<18}{sys_t:>8.0f}{info_t:>11.0f}"
              f"{prompt_t:>8.0f}{out_t:>8.1f}{base:>13.0f}")

    print("\nLegend: Reg ms = per-round base-cache build (amortized across the "
          "round's asks). Prefill ms = TTFT for one ask. Decode ms / Tok µs = "
          "generation cost. Attend = tokens actually attended; Resident = tokens "
          "physically in the KV during decode.")


if __name__ == "__main__":
    main()
