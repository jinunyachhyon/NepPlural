#!/usr/bin/env python3
"""
finalize_gold.py  —  NepPlural Step 3 (finalize)

Merges the adjudicated tie rows back into gold_labels.csv to produce the FINAL
human gold standard.

Workflow:
  1. score_annotations.py wrote gold_labels.csv (ties left blank) and
     disagreements.csv (the tie rows, with per-annotator columns).
  2. You adjudicate: in disagreements.csv, fill the axis column(s) that were tied
     (intent / primary_driver / value_orientation / affect) with the resolved label.
     Leave already-decided axes as they are.
  3. This script writes gold_final.csv, taking the adjudicated value wherever a tie
     existed and the majority value everywhere else. It validates every label and
     refuses to finalize if any tie is still unresolved (so you can't ship blanks).

Usage:
  python src/analysis/finalize_gold.py \
      --gold data/pools/sample/step3/gold_labels.csv \
      --adjudicated data/pools/sample/step3/disagreements.csv \
      --out data/pools/sample/step3/gold_final.csv
"""

import argparse
import numpy as np
import pandas as pd

AXES = ["intent", "primary_driver", "value_orientation", "affect"]
NP = "No-Persona"
ALLOWED = {
    "intent": {"Pro-Migration", "Anti-Migration", "Trapped/Regretful", "Neutral/Observation", NP},
    "primary_driver": {"Economic Necessity", "Family Obligation", "Systemic/Political Anger", "Patriotism/Love", NP},
    "value_orientation": {"Collectivist-Family", "Collectivist-Nation", "Individualist-Self", NP},
    "affect": {"Despairing/Sad", "Angry/Frustrated", "Hopeful/Motivated", "Pragmatic", NP},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/pools/sample/step3/gold_labels.csv")
    ap.add_argument("--adjudicated", default="data/pools/sample/step3/disagreements.csv")
    ap.add_argument("--out", default="data/pools/sample/step3/gold_final.csv")
    args = ap.parse_args()

    gold = pd.read_csv(args.gold).set_index("comment_id")
    adj = pd.read_csv(args.adjudicated).set_index("comment_id")

    unresolved, invalid, applied = [], [], 0
    for cidv, row in gold.iterrows():
        for ax in AXES:
            tie = bool(row.get(f"{ax}_tie", False))
            if not tie:
                continue
            # pull the adjudicated value for this tied cell
            val = adj.at[cidv, ax] if (cidv in adj.index and ax in adj.columns) else np.nan
            val = None if (isinstance(val, float) and np.isnan(val)) else (str(val).strip() if val is not None else None)
            if not val:
                unresolved.append((cidv, ax)); continue
            if val not in ALLOWED[ax]:
                invalid.append((cidv, ax, val)); continue
            gold.at[cidv, ax] = val
            applied += 1

    if invalid:
        print("INVALID adjudicated labels — fix these, then re-run:")
        for c, ax, v in invalid:
            print(f"  {c}  {ax}  -> '{v}'  (not an allowed value)")
    if unresolved:
        print(f"\nSTILL UNRESOLVED: {len(unresolved)} tied cells have no adjudicated label.")
        for c, ax in unresolved[:20]:
            print(f"  {c}  {ax}")
        if len(unresolved) > 20:
            print(f"  ... and {len(unresolved)-20} more")
    if invalid or unresolved:
        print("\nNOT writing gold_final.csv until every tie is resolved with a valid label.")
        return

    # recompute No-Persona flag and drop the helper _tie columns
    gold["is_no_persona_gold"] = (gold[AXES] == NP).all(axis=1)
    gold = gold.drop(columns=[c for c in gold.columns if c.endswith("_tie")])
    gold.to_csv(args.out)
    print(f"Applied {applied} adjudicated labels.")
    print(f"Wrote {args.out}  ({len(gold)} rows, 0 unresolved).")
    print("\nFinal gold label distribution:")
    for ax in AXES:
        print(f"\n[{ax}]")
        print(gold[ax].value_counts().to_string())


if __name__ == "__main__":
    main()
