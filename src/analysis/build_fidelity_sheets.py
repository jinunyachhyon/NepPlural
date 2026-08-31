#!/usr/bin/env python3
"""
build_fidelity_sheets.py  —  NepPlural Step 5 (fidelity annotation)

Turns the filled generation_sheet into BLIND annotation sheets: the annotators
re-classify each generated comment on the same 4 axes (targets hidden, order
shuffled), exactly as they did for real comments. Fidelity is then measured by
comparing their consensus to the hidden target.

Inputs:
  data/steerability/generation_sheet.csv   (108 rows, response filled)
Outputs (data/steerability/fidelity/):
  fidelity_sheet_A.csv  B  C   blank 4-axis sheets (identical, shuffled), no targets
  fidelity_key.csv             gen_id -> hidden targets + model/persona (YOUR analysis only)
"""

import argparse
from pathlib import Path
import pandas as pd

AXES = ["intent", "primary_driver", "value_orientation", "affect"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/steerability/generation_sheet.csv")
    ap.add_argument("--n_annotators", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="data/steerability/fidelity")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    df = df[df["response"].notna()].copy()

    # shuffle so persona/model order can't be inferred
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # KEY (hidden targets) — for scoring only, never given to annotators
    key_cols = ["gen_id", "model", "model_version", "congruence", "persona_id",
                "question_id"] + [f"target_{a}" for a in AXES]
    df[key_cols].to_csv(f"{args.out_dir}/fidelity_key.csv", index=False)

    # BLIND sheet — annotators see only the generated text
    blank = df[["gen_id", "response"]].rename(columns={"response": "comment"}).copy()
    for a in AXES:
        blank[a] = ""
    blank["is_no_persona"] = ""
    blank["notes"] = ""
    for i in range(args.n_annotators):
        name = chr(ord("A") + i)
        blank.to_csv(f"{args.out_dir}/fidelity_sheet_{name}.csv", index=False)

    print(f"wrote {args.n_annotators} blind fidelity sheets + fidelity_key.csv")
    print(f"  {len(df)} generations to judge, shuffled, targets hidden")
    print(f"  annotators classify the same 4 axes as before (+ is_no_persona)")


if __name__ == "__main__":
    main()
