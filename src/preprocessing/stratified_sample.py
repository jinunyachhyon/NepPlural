#!/usr/bin/env python3
"""
stratified_sample.py

Draw a ~300-comment sample from the substantive pool for human annotation.

Strategy ("mostly representative" with light protection):
  - Channel allocation is near-proportional to pool size, but the smallest
    channel (Thaha) gets a floor so it is not crushed by IDS.
  - Within each channel, script mix (devanagari / roman_nepali / latin / code_mixed)
    is preserved proportionally.
  - A LIGHT rare-class floor (default 15) is applied ONLY to cells that would
    otherwise fall below it, so the rarest personas (e.g. Trapped/Regretful) are
    measurable. Everything else mirrors the true distribution.
  - Disagreement is NOT forced: 98% of the pool is already non-unanimous, so a
    representative draw is naturally disagreement-rich.

Inputs : data/pools/_pool_with_llm.csv   (pool joined to old LLM labels + agreement)
Outputs (data/pools/sample/):
  sample_master.csv        -> FOR YOUR ANALYSIS ONLY. Has LLM labels + agreement.
                              Do NOT give this to annotators (it would prime them).
  annotation_sheet_A.csv   -> blank sheet for annotator A (identical items to B)
  annotation_sheet_B.csv   -> blank sheet for annotator B
  annotation_instructions.md
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

AXES = ["intent", "primary_driver", "value_orientation", "affect"]

ALLOWED = {
    "intent": ["Pro-Migration", "Anti-Migration", "Trapped/Regretful", "Neutral/Observation"],
    "primary_driver": ["Economic Necessity", "Family Obligation", "Systemic/Political Anger", "Patriotism/Love"],
    "value_orientation": ["Collectivist-Family", "Collectivist-Nation", "Individualist-Self"],
    "affect": ["Despairing/Sad", "Angry/Frustrated", "Hopeful/Motivated", "Pragmatic"],
}

# the rarest cell per axis — protected by the light floor
RARE = {
    "intent": "Trapped/Regretful",
    "value_orientation": "Collectivist-Family",
    "primary_driver": "Family Obligation",
    "affect": "Angry/Frustrated",
}


def allocate_channels(pool, target, thaha_floor):
    counts = pool.source_channel.value_counts()
    total = counts.sum()
    quota = {ch: int(round(target * n / total)) for ch, n in counts.items()}
    # apply Thaha floor by borrowing from the largest channel
    for ch in list(quota):
        if "thaha" in ch.lower() and quota[ch] < thaha_floor:
            deficit = thaha_floor - quota[ch]
            biggest = counts.idxmax()
            quota[ch] = thaha_floor
            quota[biggest] -= deficit
    return quota


def pick_within_channel(df, k, rng):
    """Proportional-by-script random pick of k rows from one channel's frame."""
    if k >= len(df):
        return df.copy()
    frac = k / len(df)
    picks = []
    for _, g in df.groupby("script"):
        kk = int(round(len(g) * frac))
        kk = min(kk, len(g))
        if kk > 0:
            picks.append(g.sample(kk, random_state=rng.integers(1e9)))
    out = pd.concat(picks) if picks else df.sample(0)
    # trim/pad to exactly k
    if len(out) > k:
        out = out.sample(k, random_state=rng.integers(1e9))
    elif len(out) < k:
        extra = df.drop(out.index)
        out = pd.concat([out, extra.sample(min(k - len(out), len(extra)),
                                           random_state=rng.integers(1e9))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/pools/_pool_with_llm.csv")
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--thaha_floor", type=int, default=40)
    ap.add_argument("--rare_floor", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_annotators", type=int, default=3,
                    help="how many identical blank sheets to emit (A, B, C, ...)")
    ap.add_argument("--out_dir", default="data/pools/sample")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pool = pd.read_csv(args.in_csv)
    pool = pool.dropna(subset=["comment"]).drop_duplicates("comment_id")

    selected_ids = set()

    # 1) light rare-class floor: only top up cells below the floor
    for ax, lab in RARE.items():
        have = pool[pool[f"llm_{ax}"] == lab]
        need = max(0, args.rare_floor - len(have.index.intersection(
            pool[pool.comment_id.isin(selected_ids)].index)))
        take = have[~have.comment_id.isin(selected_ids)]
        if len(take) > 0 and need > 0:
            forced = take.sample(min(need, len(take)), random_state=rng.integers(1e9))
            selected_ids |= set(forced.comment_id)

    # 2) proportional channel fill for the remainder
    quota = allocate_channels(pool, args.target, args.thaha_floor)
    for ch, k in quota.items():
        chdf = pool[pool.source_channel == ch]
        already = chdf[chdf.comment_id.isin(selected_ids)]
        remaining = k - len(already)
        rest = chdf[~chdf.comment_id.isin(selected_ids)]
        if remaining > 0 and len(rest) > 0:
            picks = pick_within_channel(rest, remaining, rng)
            selected_ids |= set(picks.comment_id)

    sample = pool[pool.comment_id.isin(selected_ids)].copy()
    # trim to target if the floor pushed us over
    if len(sample) > args.target:
        # never drop the forced rare rows: keep them, trim from the rest
        rare_mask = np.zeros(len(sample), dtype=bool)
        for ax, lab in RARE.items():
            rare_mask |= (sample[f"llm_{ax}"] == lab).values
        rare = sample[rare_mask]
        rest = sample[~rare_mask].sample(max(0, args.target - len(rare)),
                                         random_state=rng.integers(1e9))
        sample = pd.concat([rare, rest])

    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)  # shuffle order

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # master (with LLM labels) — analysis only
    master_cols = (["comment_id", "comment", "source_channel", "source_file",
                    "script", "word_count"]
                   + [f"llm_{a}" for a in AXES] + [f"agree_{a}" for a in AXES]
                   + ["n_disagree_axes"])
    master = sample[[c for c in master_cols if c in sample.columns]]
    master.to_csv(f"{args.out_dir}/sample_master.csv", index=False)

    # blank annotation sheets (NO LLM labels) — one identical sheet per annotator
    blank = sample[["comment_id", "comment", "source_channel", "script"]].copy()
    for ax in AXES:
        blank[ax] = ""
    blank["is_no_persona"] = ""
    blank["notes"] = ""
    sheet_names = [chr(ord("A") + i) for i in range(args.n_annotators)]  # A, B, C, ...
    for name in sheet_names:
        blank.to_csv(f"{args.out_dir}/annotation_sheet_{name}.csv", index=False)

    # instructions
    lines = ["# NepPlural annotation — allowed values\n",
             "Label every comment on all four axes. If a comment expresses no migration",
             "persona (off-topic / pure praise), tick `is_no_persona` = 1 and leave the axes blank.",
             "Work independently — do not discuss individual comments with the other annotator.\n"]
    for ax in AXES:
        lines.append(f"**{ax}**: " + " | ".join(ALLOWED[ax]))
    lines.append("\n`is_no_persona`: 1 if off-topic/no persona, else 0")
    lines.append("`notes`: optional free text for hard cases")
    lines.append("\nFinal labels are set by majority vote across annotators (2-of-3).")
    lines.append("Do not edit, sort, or reorder the `comment_id` column — it is the merge key.")
    Path(f"{args.out_dir}/annotation_instructions.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- report
    print(f"SAMPLE SIZE: {len(sample)}  (target {args.target})\n")
    print("channel allocation:")
    print(sample.source_channel.value_counts().to_string())
    print("\nscript mix:")
    print(sample.script.value_counts().to_string())
    print("\nrare-cell coverage (by LLM prediction) in sample:")
    for ax, lab in RARE.items():
        print(f"  {ax}={lab}: {(sample[f'llm_{ax}']==lab).sum()}")
    print("\nrepresentativeness check (sample share vs pool share, LLM-predicted):")
    for ax in AXES:
        s = sample[f"llm_{ax}"].value_counts(normalize=True)
        p = pool[f"llm_{ax}"].value_counts(normalize=True)
        print(f"  [{ax}]")
        for lab in ALLOWED[ax]:
            print(f"     {lab:22s} sample {100*s.get(lab,0):5.1f}%  pool {100*p.get(lab,0):5.1f}%")
    print("\ndisagreement in sample (n axes not unanimous):")
    print(sample.n_disagree_axes.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
