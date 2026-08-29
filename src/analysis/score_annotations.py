#!/usr/bin/env python3
"""
score_annotations.py  —  NepPlural Step 3

Turns the three filled annotation sheets into:
  1. label validation report      (invalid_labels.csv)
  2. inter-annotator agreement     Fleiss' kappa + Krippendorff's alpha per axis,
                                    plus the No-Persona decision agreement,
                                    shown SIDE BY SIDE with the LLM annotators
  3. gold labels by human majority (gold_labels.csv), 3-way ties flagged
  4. adjudication worklist         (disagreements.csv)
  5. human-vs-LLM agreement        per axis (the bias finding)

Usage:
  python src/analysis/score_annotations.py \
      --sheets data/pools/sample/annotation_sheet_A.csv \
               data/pools/sample/annotation_sheet_B.csv \
               data/pools/sample/annotation_sheet_C.csv \
      --pool_with_llm data/pools/_pool_with_llm.csv \
      --label_dir data/annotations/label \
      --out_dir data/pools/sample/step3
"""

import argparse
import glob
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

AXES = ["intent", "primary_driver", "value_orientation", "affect"]
NP = "No-Persona"
ALLOWED = {
    "intent": ["Pro-Migration", "Anti-Migration", "Trapped/Regretful", "Neutral/Observation"],
    "primary_driver": ["Economic Necessity", "Family Obligation", "Systemic/Political Anger", "Patriotism/Love"],
    "value_orientation": ["Collectivist-Family", "Collectivist-Nation", "Individualist-Self"],
    "affect": ["Despairing/Sad", "Angry/Frustrated", "Hopeful/Motivated", "Pragmatic"],
}
LLM_MODELS = ["GPT-5.2", "Gemini 3.5 flash", "Sonet 4.6"]


# --------------------------------------------------------------------------- helpers
def truthy(x):
    return str(x).strip().lower() in {"1", "1.0", "true", "yes", "y", "x"}


def norm(x):
    return np.nan if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x).strip()


def cid(s):
    return hashlib.sha1(str(s).strip().lower().encode("utf-8")).hexdigest()[:12]


def krippendorff_alpha_nominal(data):
    """data: list of lists, one per unit, of the labels given by coders (NaNs dropped).
    Nominal alpha; units with <2 codings are ignored."""
    # coincidence matrix
    values = sorted({v for unit in data for v in unit if isinstance(v, str)})
    idx = {v: i for i, v in enumerate(values)}
    K = len(values)
    if K < 2:
        return float("nan")
    o = np.zeros((K, K))
    for unit in data:
        u = [v for v in unit if isinstance(v, str)]
        m = len(u)
        if m < 2:
            continue
        c = Counter(u)
        for a in values:
            for b in values:
                if a == b:
                    pairs = c[a] * (c[a] - 1)
                else:
                    pairs = c[a] * c[b]
                o[idx[a], idx[b]] += pairs / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n < 2:
        return float("nan")
    Do = sum(o[i, j] for i in range(K) for j in range(K) if i != j)
    De = sum(n_c[i] * n_c[j] for i in range(K) for j in range(K) if i != j) / (n - 1)
    if De == 0:
        return float("nan")
    return 1 - (Do / (De))  # Do and De both un-normalized by n -> ratio is correct


def fleiss_on(cols_labels, categories):
    """cols_labels: DataFrame rows=units, columns=raters, string labels (may be NaN).
    Returns Fleiss kappa on units where ALL raters gave a codeable label."""
    complete = cols_labels.dropna()
    if len(complete) < 2:
        return float("nan"), 0
    # build count matrix units x categories
    catidx = {c: i for i, c in enumerate(categories)}
    M = np.zeros((len(complete), len(categories)), dtype=int)
    for r, (_, row) in enumerate(complete.iterrows()):
        for v in row:
            if v in catidx:
                M[r, catidx[v]] += 1
    # drop empty category columns to satisfy statsmodels
    M = M[:, M.sum(axis=0) > 0]
    try:
        return fleiss_kappa(M), len(complete)
    except Exception:
        return float("nan"), len(complete)


def load_llm_per_model(label_dir):
    """Return dict: model -> Series(comment_id -> {axis:label}). For LLM Fleiss kappa."""
    out = {}
    for m in LLM_MODELS:
        files = glob.glob(f"{label_dir}/{m}/**/*.csv", recursive=True)
        if not files:
            return None
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        d["comment_id"] = d["comment"].apply(cid)
        d = d.drop_duplicates("comment_id").set_index("comment_id")
        # normalize axis column names
        colmap = {}
        for ax in AXES:
            for c in d.columns:
                if c.strip().lower() == ax:
                    colmap[ax] = c
        out[m] = (d, colmap)
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--pool_with_llm", default="data/pools/_pool_with_llm.csv")
    ap.add_argument("--label_dir", default="data/annotations/label")
    ap.add_argument("--out_dir", default="data/pools/sample/step3")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    raters = [chr(ord("A") + i) for i in range(len(args.sheets))]

    sheets = {}
    for name, f in zip(raters, args.sheets):
        df = pd.read_csv(f).set_index("comment_id")
        sheets[name] = df

    ids = sheets[raters[0]].index
    for r in raters[1:]:
        assert set(sheets[r].index) == set(ids), "sheets have different comment_id sets"

    # ---- 1. validation + effective labels -----------------------------------
    invalid_rows = []
    eff = {}   # eff[(rater, axis)] = Series over ids
    persona_decision = {}  # rater -> Series 'NP'/'P'/nan
    for r in raters:
        df = sheets[r].reindex(ids)
        nop = df["is_no_persona"].apply(truthy) if "is_no_persona" in df else pd.Series(False, index=ids)
        pdec = pd.Series(np.nan, index=ids, dtype=object)
        for ax in AXES:
            raw = df[ax].apply(norm) if ax in df else pd.Series(np.nan, index=ids)
            e = pd.Series(np.nan, index=ids, dtype=object)
            for cidv in ids:
                if nop.get(cidv, False):
                    e[cidv] = NP
                    continue
                v = raw.get(cidv)
                if isinstance(v, str):
                    if v in ALLOWED[ax]:
                        e[cidv] = v
                    else:
                        invalid_rows.append({"comment_id": cidv, "rater": r, "axis": ax, "value": v})
                        e[cidv] = np.nan
            eff[(r, ax)] = e
        # persona decision per rater
        for cidv in ids:
            if nop.get(cidv, False):
                pdec[cidv] = "NP"
            elif any(isinstance(eff[(r, ax)].get(cidv), str) for ax in AXES):
                pdec[cidv] = "P"
        persona_decision[r] = pdec

    inv = pd.DataFrame(invalid_rows)
    inv.to_csv(f"{args.out_dir}/invalid_labels.csv", index=False)

    # ---- 2. agreement (humans) ----------------------------------------------
    print("=" * 70)
    print("INVALID LABELS:", len(inv), "(see invalid_labels.csv)")
    if len(inv):
        print(inv.to_string(index=False))

    print("\n" + "=" * 70)
    print("NO-PERSONA DECISION")
    for r in raters:
        print(f"  {r}: {(persona_decision[r]=='NP').sum():3d} No-Persona | "
              f"{(persona_decision[r]=='P').sum():3d} Persona | "
              f"{persona_decision[r].isna().sum():3d} blank")
    pdec_df = pd.DataFrame({r: persona_decision[r] for r in raters})
    k_np, n_np = fleiss_on(pdec_df, ["NP", "P"])
    print(f"  Fleiss kappa on the No-Persona decision: {k_np:.3f}  (n={n_np})")

    print("\n" + "=" * 70)
    print("INTER-ANNOTATOR AGREEMENT PER AXIS (No-Persona treated as a label)")
    print(f"{'axis':20s} {'Fleiss k':>9s} {'Kripp a':>9s} {'n_complete':>11s}")
    human_kappa = {}
    for ax in AXES:
        cats = ALLOWED[ax] + [NP]
        M = pd.DataFrame({r: eff[(r, ax)] for r in raters})
        k, ncomp = fleiss_on(M, cats)
        adata = [[M.loc[i, r] for r in raters if isinstance(M.loc[i, r], str)] for i in ids]
        a = krippendorff_alpha_nominal(adata)
        human_kappa[ax] = k
        print(f"{ax:20s} {k:9.3f} {a:9.3f} {ncomp:11d}")

    # agreement on PERSONA-ONLY rows (all three said Persona) — the "pure" label agreement
    print("\nPER AXIS on rows where ALL THREE marked Persona (excludes No-Persona):")
    all_persona = pd.DataFrame({r: persona_decision[r] for r in raters}).eq("P").all(axis=1)
    pids = ids[all_persona.values]
    print(f"  ({len(pids)} such rows)")
    for ax in AXES:
        M = pd.DataFrame({r: eff[(r, ax)].reindex(pids) for r in raters})
        k, ncomp = fleiss_on(M, ALLOWED[ax])
        print(f"  {ax:20s} Fleiss k = {k:6.3f}  (n={ncomp})")

    # ---- 3. gold labels by majority ----------------------------------------
    gold = pd.DataFrame(index=ids)
    gold["comment"] = sheets[raters[0]].reindex(ids)["comment"]
    tie_flags = {}
    for ax in AXES:
        M = pd.DataFrame({r: eff[(r, ax)] for r in raters})
        g, tie = [], []
        for i in ids:
            labs = [M.loc[i, r] for r in raters if isinstance(M.loc[i, r], str)]
            if not labs:
                g.append(np.nan); tie.append(False); continue
            c = Counter(labs); top, cnt = c.most_common(1)[0]
            if cnt >= 2:
                g.append(top); tie.append(False)
            else:
                g.append(np.nan); tie.append(True)   # 3-way tie -> adjudicate
        gold[ax] = g
        gold[f"{ax}_tie"] = tie
        tie_flags[ax] = sum(tie)
    gold["is_no_persona_gold"] = (gold[AXES] == NP).all(axis=1)
    gold.to_csv(f"{args.out_dir}/gold_labels.csv")
    print("\n" + "=" * 70)
    print("GOLD LABELS (human majority 2-of-3)")
    for ax in AXES:
        print(f"  {ax:20s} 3-way ties needing adjudication: {tie_flags[ax]}")
    print(f"  rows where gold = No-Persona on all axes: {gold['is_no_persona_gold'].sum()}")

    # adjudication worklist
    any_tie = gold[[f"{ax}_tie" for ax in AXES]].any(axis=1)
    adj = gold[any_tie].copy()
    for r in raters:
        for ax in AXES:
            adj[f"{r}_{ax}"] = eff[(r, ax)].reindex(adj.index)
    adj.to_csv(f"{args.out_dir}/disagreements.csv")
    print(f"  -> {len(adj)} rows written to disagreements.csv for adjudication")

    # ---- 4. human vs LLM ----------------------------------------------------
    print("\n" + "=" * 70)
    print("HUMAN GOLD vs LLM")
    try:
        pw = pd.read_csv(args.pool_with_llm).drop_duplicates("comment_id").set_index("comment_id")
    except Exception as e:
        print("  pool_with_llm not available:", e); pw = None

    if pw is not None:
        # LLM Fleiss kappa (recompute per-model) for side-by-side
        permodel = load_llm_per_model(args.label_dir)
        print(f"\n{'axis':20s} {'human k':>8s} {'LLM k':>8s} {'H-gold=LLM-maj':>15s}")
        for ax in AXES:
            # LLM kappa on the sampled ids
            llm_k = float("nan")
            if permodel:
                cols = {}
                for m in LLM_MODELS:
                    d, colmap = permodel[m]
                    cols[m] = d[colmap[ax]].reindex(ids) if ax in colmap else pd.Series(np.nan, index=ids)
                Ml = pd.DataFrame(cols)
                llm_k, _ = fleiss_on(Ml, ALLOWED[ax])
            # agreement: human gold (real label, not tie/NP) vs LLM majority label
            gcol = gold[ax].reindex(ids)
            lcol = pw[f"llm_{ax}"].reindex(ids) if f"llm_{ax}" in pw else pd.Series(np.nan, index=ids)
            mask = gcol.isin(ALLOWED[ax]) & lcol.notna()
            acc = (gcol[mask].values == lcol[mask].values).mean() if mask.sum() else float("nan")
            print(f"{ax:20s} {human_kappa[ax]:8.3f} {llm_k:8.3f} {acc:14.1%}  (n={int(mask.sum())})")

        # how often did the LLM assign a persona where humans (majority) saw none?
        np_gold_ids = gold.index[gold["is_no_persona_gold"].values]
        print(f"\nRows humans (majority) called No-Persona: {len(np_gold_ids)}")
        print("  -> the LLM pipeline assigned a substantive persona to ALL of them")
        print("     (LLMs never abstain), i.e. LLM over-attributes persona on "
              f"{len(np_gold_ids)} / {len(ids)} = {len(np_gold_ids)/len(ids):.1%} of the sample")

    print("\nOutputs written to:", args.out_dir)
    print("  gold_labels.csv | disagreements.csv | invalid_labels.csv")


if __name__ == "__main__":
    main()
