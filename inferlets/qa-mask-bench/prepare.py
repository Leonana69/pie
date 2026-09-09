#!/usr/bin/env python3
"""Generate a deterministic synthetic QA dataset for the masking experiment.

Each *item* is a self-contained fictional fact about a uniquely named
institution, plus a single-hop question whose answer appears in ONLY that item.
Because the names/values are fictional and unique, the model cannot answer from
prior knowledge — it must attend to the right document. That makes accuracy a
sharp test of whether masking/dropping kept the right info reachable.

Items are grouped into *rounds* of `--n-paragraphs` items (default 6). For each
round we pick `--asks-per-round` (default 5) random pairs of questions out of the
C(6,2)=15 possible pairs — each ask poses two questions whose two infos sit at
random positions among the cached six.

Usage:
  python prepare.py --n-rounds 5 --n-paragraphs 6 --asks-per-round 5 --seed 42
"""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROUNDS_PATH = HERE / "rounds.json"

# Fictional building blocks — combined to make unique, non-real proper nouns.
PRE = [
    "Vel", "Quor", "Thal", "Mor", "Zeph", "Bryn", "Cal", "Drav", "Eld", "Fenn",
    "Grim", "Hael", "Ixa", "Jor", "Kael", "Lum", "Myr", "Nyx", "Oth", "Pyr",
    "Quill", "Rho", "Syl", "Tor", "Umbra", "Vex", "Wyn", "Xan", "Yor", "Zan",
    "Aer", "Bel", "Cyr", "Dor", "Esk", "Fal", "Gan", "Hod", "Irn", "Jav",
]
SUF = [
    "stone", "hollow", "mere", "grove", "port", "reach", "fall", "gate",
    "spire", "haven", "ridge", "vale", "cross", "mont", "burgh", "wick",
    "ford", "helm", "barrow", "crest", "marsh", "thorpe", "garde", "dell",
]
INST_TYPE = [
    "Archive", "Observatory", "Institute", "Conservatory", "Foundry",
    "Repository", "Gallery", "Academy", "Atheneum", "Lyceum",
]
FIRST = [
    "Oletta", "Hadwin", "Marlo", "Sephine", "Doran", "Yvenna", "Karsil",
    "Brielle", "Tovin", "Esria", "Galwen", "Nessa", "Pell", "Rovis",
    "Thessaly", "Ulric", "Vanwen", "Corin", "Delphine", "Ferrin",
]
LAST = [
    "Vance", "Brue", "Calder", "Drummel", "Eskin", "Farrow", "Greel",
    "Hallow", "Ivers", "Jannik", "Korr", "Lunde", "Mott", "Norr",
    "Osric", "Penhall", "Quist", "Rade", "Storne", "Tully",
]


# Neutral filler used to inflate token counts (for the --scale sweep). Contains
# no digits or proper nouns, so it never collides with an answer.
INFO_FILLER = [
    "The institution is widely regarded as a regional landmark.",
    "Its reading rooms remain open to visiting scholars throughout the year.",
    "The building is admired for its understated architecture and quiet courtyards.",
    "Staff there maintain careful indexes for the convenience of researchers.",
    "It has long served as a gathering place for the surrounding community.",
    "Seasonal exhibitions draw curious visitors from neighbouring towns.",
    "The collection is organised into several thematic wings.",
    "Volunteers assist with cataloguing and with the upkeep of the grounds.",
    "A modest cafe operates near the main entrance for the comfort of guests.",
    "The institution publishes an occasional newsletter for its patrons.",
    "Its corridors are lined with portraits of former benefactors.",
    "Researchers describe the atmosphere as calm and conducive to study.",
]
SYS_FILLER = [
    "Read every document carefully before you respond.",
    "Do not speculate beyond what the documents explicitly state.",
    "Keep each answer concise and free of extraneous commentary.",
    "If a detail is not present in the documents, do not invent it.",
    "Treat each document as an independent, self-contained source.",
    "Prefer the shortest span of text that fully answers the question.",
    "Maintain a neutral and factual tone at all times.",
    "Answer in the same language as the question.",
]


def inflate(base_text: str, pool, rng, target_words: int) -> str:
    """Append filler sentences to base_text until it reaches ~target_words."""
    words = base_text.split()
    while len(words) < target_words:
        words.extend(rng.choice(pool).split())
    return " ".join(words)


def make_unique_namer(rng):
    used = set()

    def fresh(parts_a, parts_b, sep=""):
        for _ in range(10000):
            name = rng.choice(parts_a) + sep + rng.choice(parts_b)
            if name not in used:
                used.add(name)
                return name
        raise RuntimeError("ran out of unique names — enlarge the pools")

    return fresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rounds", type=int, default=5)
    ap.add_argument("--n-paragraphs", type=int, default=6,
                    help="Items (info+question pairs) per round.")
    ap.add_argument("--asks-per-round", type=int, default=5,
                    help="Random question-pairs sampled per round (<= C(n,2)).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Inflate system + info token counts by ~this factor "
                         "using neutral filler. Core facts and asks are "
                         "unchanged (separate RNG), so scales are comparable.")
    ap.add_argument("--out", default=str(ROUNDS_PATH),
                    help="Output JSON path (default rounds.json).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # Separate stream for filler so core facts/asks are identical across scales.
    frng = random.Random(args.seed + 12345)
    name_inst = make_unique_namer(rng)
    name_person = make_unique_namer(rng)
    name_city = make_unique_namer(rng)

    n_pairs = args.n_paragraphs * (args.n_paragraphs - 1) // 2
    if args.asks_per_round > n_pairs:
        sys.stderr.write(
            f"asks-per-round ({args.asks_per_round}) > C({args.n_paragraphs},2)"
            f"={n_pairs}; clamping.\n"
        )
        args.asks_per_round = n_pairs

    # (attribute key, question template, value-format) — answer lives only in this item.
    def q_director(inst):
        return f"Who is the director of the {inst}?"

    def q_founded(inst):
        return f"In what year was the {inst} founded?"

    def q_holdings(inst):
        return f"How many catalogued artifacts does the {inst} hold?"

    def q_city(inst):
        return f"In which city is the {inst} located?"

    QVARIANTS = ["director", "founded", "holdings", "city"]

    rounds = []
    for r in range(args.n_rounds):
        items = []
        for _ in range(args.n_paragraphs):
            inst = name_inst(PRE, SUF) + " " + rng.choice(INST_TYPE)
            city = name_city(PRE, SUF)
            director = name_person(FIRST, LAST, sep=" ")
            year = rng.randint(1604, 1992)
            holdings = rng.randint(1031, 98750)

            info = (
                f"The {inst} is located in the city of {city}. "
                f"It was founded in {year} and is directed by {director}. "
                f"The {inst} holds {holdings} catalogued artifacts."
            )
            if args.scale > 1.0:
                # ~base_words words at scale 1; inflate the rest with filler.
                target = int(len(info.split()) * args.scale)
                info = inflate(info, INFO_FILLER, frng, target)

            attr = rng.choice(QVARIANTS)
            if attr == "director":
                question, answer = q_director(inst), director
            elif attr == "founded":
                question, answer = q_founded(inst), str(year)
            elif attr == "holdings":
                question, answer = q_holdings(inst), str(holdings)
            else:
                question, answer = q_city(inst), city

            items.append({
                "info": info,
                "question": question,
                "answer": answer,
                "inst": inst,
                "attr": attr,
            })

        all_pairs = list(itertools.combinations(range(args.n_paragraphs), 2))
        rng.shuffle(all_pairs)
        asks = [list(p) for p in all_pairs[: args.asks_per_round]]

        rd = {"round_id": r, "items": items, "asks": asks}
        if args.scale > 1.0:
            # Inflate the system prompt by ~(scale-1)x of its ~50-word base.
            rd["sys_extra"] = inflate("", SYS_FILLER, frng, int((args.scale - 1) * 32))
        rounds.append(rd)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(rounds, indent=2))
    n_asks = sum(len(rd["asks"]) for rd in rounds)
    print(
        f"Wrote {len(rounds)} rounds × {args.n_paragraphs} items "
        f"({len(rounds) * args.n_paragraphs} items total), "
        f"{n_asks} asks ({2 * n_asks} questions), scale={args.scale} → {out_path}",
        file=sys.stderr,
    )
    # Sanity: attribute distribution.
    dist = {}
    for rd in rounds:
        for it in rd["items"]:
            dist[it["attr"]] = dist.get(it["attr"], 0) + 1
    print(f"Attribute distribution: {dist}", file=sys.stderr)


if __name__ == "__main__":
    main()
