#!/usr/bin/env python3
"""
build_splits.py  —  NepPlural Step 4 (data assembly)

Assembles the encoder train/val/test splits for the clean, human-anchored benchmark.

  TEST  = the 300 human-gold comments (5 classes per axis, incl. No-Persona).
  TRAIN = the rest of the substantive pool, labelled by LLM majority (persona classes),
          PLUS the deterministically-routed no_persona comments (No-Persona class).
  VAL   = a stratified slice held out of TRAIN for early stopping.

Design notes:
  - This is deliberately "train on cheap LLM labels, test on human gold." The gap to
    the human ceiling measures how much LLM label noise costs.
  - No test comment can appear in train/val (checked and asserted).
  - Label space is unified to 5 classes per axis so train and test match.

Inputs:
  data/pools/sample/step3/gold_final.csv   (human gold, 300)
  data/pools/_pool_with_llm.csv            (substantive pool + LLM majority labels)
  data/pools/no_persona.csv                (routed off-topic -> No-Persona training)
Outputs (data/splits/):
  train.csv  val.csv  test.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

AXES = ["intent", "primary_driver", "value_orientation", "affect"]
NP = "No-Persona"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/pools/sample/step3/gold_final.csv")
    ap.add_argument("--pool_with_llm", default="data/pools/_pool_with_llm.csv")
    ap.add_argument("--no_persona", default="data/pools/no_persona.csv")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="data/splits")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ---- TEST: human gold -----------------------------------------------------
    gold = pd.read_csv(args.gold)
    test = gold[["comment_id", "comment"] + AXES].copy()
    test_ids = set(test.comment_id)

    # ---- TRAIN persona classes: LLM labels on substantive pool minus test -----
    pool = pd.read_csv(args.pool_with_llm)
    pool = pool[~pool.comment_id.isin(test_ids)].copy()          # no leakage
    persona = pool[["comment_id", "comment"]].copy()
    for ax in AXES:
        persona[ax] = pool[f"llm_{ax}"]
    persona = persona.dropna(subset=AXES)                        # drop rows missing any LLM label

    # ---- TRAIN No-Persona class: routed off-topic comments --------------------
    nop = pd.read_csv(args.no_persona)
    nop = nop[~nop.comment_id.isin(test_ids)].copy()
    nopdf = nop[["comment_id", "comment"]].copy()
    for ax in AXES:
        nopdf[ax] = NP

    train_all = pd.concat([persona, nopdf], ignore_index=True).drop_duplicates("comment_id")
    train_all = train_all[~train_all.comment_id.isin(test_ids)]  # belt and suspenders

    # ---- VAL: stratified slice out of train (by intent) -----------------------
    val_parts = []
    for lab, g in train_all.groupby("intent"):
        k = max(1, int(round(len(g) * args.val_frac)))
        val_parts.append(g.sample(min(k, len(g)), random_state=int(rng.integers(1e9))))
    val = pd.concat(val_parts)
    train = train_all.drop(index=val.index)

    # ---- leakage assertions ---------------------------------------------------
    assert not (set(train.comment_id) & test_ids), "LEAK: train ∩ test"
    assert not (set(val.comment_id) & test_ids), "LEAK: val ∩ test"
    assert not (set(train.comment_id) & set(val.comment_id)), "LEAK: train ∩ val"

    for name, df in [("train", train), ("val", val), ("test", test)]:
        df[["comment_id", "comment"] + AXES].to_csv(f"{args.out_dir}/{name}.csv", index=False)

    # ---- report ---------------------------------------------------------------
    print(f"train {len(train)} | val {len(val)} | test {len(test)}  (no leakage)")
    print(f"  train persona rows (LLM-labelled): {len(persona)} | No-Persona rows: {len(nopdf)}")
    for ax in AXES:
        print(f"\n[{ax}]  test (human gold) class support:")
        print(test[ax].value_counts().to_string())
    print("\nNOTE: rare test classes (support < ~8) are reported but per-class F1 on them")
    print("      is high-variance — lead with macro-F1 and state the caveat.")


if __name__ == "__main__":
    main()
