#!/usr/bin/env python3
"""
join_llm_labels.py

Builds data/pools/_pool_with_llm.csv : the substantive pool joined to the OLD
three-model LLM labels, with a per-axis majority label and an agreement count.
This file is the input to stratified_sample.py.

The old LLM labels are used ONLY as a sampling signal (to protect rare cells and
to locate disagreement), never as ground truth.

Inputs:
  data/pools/substantive_pool.csv                     (from prepare_pools.py)
  data/annotations/label/<model>/**/*.csv             (per-model LLM labels)
Output:
  data/pools/_pool_with_llm.csv
"""

import glob
import hashlib
from collections import Counter
import pandas as pd

AXES = ["intent", "primary_driver", "value_orientation", "affect"]
MODELS = ["GPT-5.2", "Gemini 3.5 flash", "Sonet 4.6"]   # folder names under data/annotations/label/


def cid(s):
    return hashlib.sha1(str(s).strip().lower().encode("utf-8")).hexdigest()[:12]


def axis_col(df, ax):
    for c in df.columns:
        if c.strip().lower() == ax:
            return c
    return None


def load_model(m):
    frames = []
    for f in glob.glob(f"data/annotations/label/{m}/**/*.csv", recursive=True):
        frames.append(pd.read_csv(f))
    if not frames:
        raise FileNotFoundError(f"no label CSVs found for model '{m}' under data/annotations/label/{m}/")
    d = pd.concat(frames, ignore_index=True)
    d["comment_id"] = d["comment"].apply(cid)
    return d.drop_duplicates("comment_id").set_index("comment_id")


def main():
    pool = pd.read_csv("data/pools/substantive_pool.csv")
    md = {m: load_model(m) for m in MODELS}

    # ids present in all three models
    ids = set(md[MODELS[0]].index)
    for m in MODELS[1:]:
        ids &= set(md[m].index)

    recs = []
    for cidv in ids:
        r = {"comment_id": cidv}
        for ax in AXES:
            labs = []
            for m in MODELS:
                col = axis_col(md[m], ax)
                v = md[m].at[cidv, col] if col in md[m].columns else None
                if isinstance(v, str):
                    labs.append(v)
            if labs:
                top, n = Counter(labs).most_common(1)[0]
                r[f"llm_{ax}"] = top
                r[f"agree_{ax}"] = n            # 3=unanimous, 2=majority, 1=three-way tie
            else:
                r[f"llm_{ax}"] = None
                r[f"agree_{ax}"] = 0
        recs.append(r)

    llm = pd.DataFrame(recs)
    m = pool.merge(llm, on="comment_id", how="left")
    m["n_disagree_axes"] = (m[[f"agree_{ax}" for ax in AXES]] < 3).sum(axis=1)

    cov = m["llm_intent"].notna().mean()
    m.to_csv("data/pools/_pool_with_llm.csv", index=False)
    print(f"wrote data/pools/_pool_with_llm.csv  ({len(m)} rows)")
    print(f"join coverage (pool rows with LLM labels): {100*cov:.1f}%")


if __name__ == "__main__":
    main()
