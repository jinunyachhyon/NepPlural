#!/usr/bin/env python3
"""
human_ceiling.py  —  NepPlural Step 4 (reference line)

Computes the HUMAN CEILING: how well an individual human annotator predicts the
adjudicated gold label, per axis. This is the reference an encoder's macro-F1
should be read against — "0.52 out of a human ceiling of X", not "0.52 out of 1.0".

For each annotator we treat their labels as predictions and the adjudicated gold
as truth, then average macro-F1 and accuracy across annotators. No-Persona is
included as a label (an annotator's is_no_persona flag -> "No-Persona").

Usage:
  python src/analysis/human_ceiling.py \
      --sheets data/pools/sample/annotation_sheet_A.csv \
               data/pools/sample/annotation_sheet_B.csv \
               data/pools/sample/annotation_sheet_C.csv \
      --gold data/pools/sample/step3/gold_final.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

AXES = ["intent", "primary_driver", "value_orientation", "affect"]
NP = "No-Persona"


def truthy(x):
    return str(x).strip().lower() in {"1", "1.0", "true", "yes", "y", "x"}


def eff_label(row, ax):
    if "is_no_persona" in row and truthy(row["is_no_persona"]):
        return NP
    v = row.get(ax)
    return np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--gold", default="data/pools/sample/step3/gold_final.csv")
    args = ap.parse_args()

    gold = pd.read_csv(args.gold).set_index("comment_id")
    raters = [chr(ord("A") + i) for i in range(len(args.sheets))]

    print(f"{'axis':20s} {'mean macro-F1':>14s} {'mean accuracy':>14s}")
    axis_macro = {}
    for ax in AXES:
        f1s, accs = [], []
        for name, f in zip(raters, args.sheets):
            s = pd.read_csv(f).set_index("comment_id")
            common = gold.index.intersection(s.index)
            y_true = gold.loc[common, ax].astype(str).values
            y_pred = np.array([eff_label(s.loc[c], ax) for c in common], dtype=object)
            mask = pd.notna(y_pred)
            yt, yp = y_true[mask], y_pred[mask].astype(str)
            f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
            accs.append(accuracy_score(yt, yp))
        axis_macro[ax] = np.mean(f1s)
        print(f"{ax:20s} {np.mean(f1s):14.3f} {np.mean(accs):14.3f}")
    print(f"\n{'MEAN across axes':20s} {np.mean(list(axis_macro.values())):14.3f}")
    print("\nRead your encoder macro-F1 against these ceilings, not against 1.0.")


if __name__ == "__main__":
    main()
