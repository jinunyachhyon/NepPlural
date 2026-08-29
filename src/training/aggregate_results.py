#!/usr/bin/env python3
"""
aggregate_results.py  —  NepPlural Step 4 (final table)

Reads the per-seed test predictions written by train_multitask.py and produces
the benchmark's headline results against the human-gold test set:

  - per-model mean +/- std macro-F1 (per axis + overall) across seeds
  - per-class F1 and TEST SUPPORT (so tiny classes are visible, not hidden)
  - a compact leaderboard sorted by overall mean macro-F1

The test labels in the prediction files ARE the human gold (build_splits put the
300 gold comments there), so this is a direct model-vs-human comparison.

Usage:
  python src/training/aggregate_results.py --results_dir src/training/Results_v2
"""

import argparse, glob, os, json
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TASKS = ["intent", "primary_driver", "value_orientation", "affect"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="src/training/Results_v2")
    ap.add_argument("--out_dir", default="src/training/Results_v2")
    args = ap.parse_args()

    model_dirs = [d for d in glob.glob(f"{args.results_dir}/*") if os.path.isdir(d)]
    if not model_dirs:
        print(f"No model folders under {args.results_dir}. Run train_multitask.py first.")
        return

    leaderboard = []
    per_class_rows = []
    for md in sorted(model_dirs):
        model = os.path.basename(md).replace("__", "/")
        pred_files = sorted(glob.glob(f"{md}/seed*_test_predictions.csv"))
        if not pred_files:
            continue
        # per-seed macro-F1 per axis
        axis_scores = defaultdict(list)
        # per-class F1 accumulation (across seeds) + support from first file
        support = {}
        classf1 = defaultdict(lambda: defaultdict(list))
        for pf in pred_files:
            df = pd.read_csv(pf)
            for t in TASKS:
                y_true = df[t].astype(str)
                y_pred = df[f"{t}_pred"].astype(str)
                axis_scores[t].append(f1_score(y_true, y_pred, average="macro", zero_division=0))
                labels = sorted(y_true.unique())
                per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
                for lab, v in zip(labels, per):
                    classf1[(t, lab)][pf].append(v)
                if t not in support:
                    support[t] = y_true.value_counts().to_dict()

        row = {"model": model, "n_seeds": len(pred_files)}
        means = []
        for t in TASKS:
            m, s = np.mean(axis_scores[t]), np.std(axis_scores[t])
            row[t] = f"{m:.3f}±{s:.3f}"
            means.append(m)
        row["mean_macro_f1"] = np.mean(means)
        leaderboard.append(row)

        for (t, lab), d in classf1.items():
            vals = [v for lst in d.values() for v in lst]
            per_class_rows.append({"model": model, "axis": t, "class": lab,
                                   "support": support[t].get(lab, 0),
                                   "f1_mean": round(np.mean(vals), 3),
                                   "f1_std": round(np.std(vals), 3)})

    lb = pd.DataFrame(leaderboard).sort_values("mean_macro_f1", ascending=False)
    lb_display = lb.copy()
    lb_display["mean_macro_f1"] = lb_display["mean_macro_f1"].map(lambda x: f"{x:.3f}")
    os.makedirs(args.out_dir, exist_ok=True)
    lb.to_csv(f"{args.out_dir}/leaderboard.csv", index=False)
    pcf = pd.DataFrame(per_class_rows).sort_values(["model", "axis", "support"], ascending=[True, True, False])
    pcf.to_csv(f"{args.out_dir}/per_class_f1.csv", index=False)

    print("=" * 70)
    print("LEADERBOARD  (test = human gold; mean±std macro-F1 over seeds)")
    print("=" * 70)
    print(lb_display.to_string(index=False))
    print("\nHuman ceiling (from human_ceiling.py): mean macro-F1 ~0.80")
    print("Read every model score against that ceiling, not against 1.0.\n")
    print("Per-class F1 + support written to per_class_f1.csv.")
    print("Rare classes (support < ~20) are high-variance — cite macro-F1, caveat per-class.")


if __name__ == "__main__":
    main()
