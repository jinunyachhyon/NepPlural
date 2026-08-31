#!/usr/bin/env python3
"""
score_steerability.py  —  NepPlural Step 5 (results)

Computes the headline steerability results from the judged fidelity sheets.

Fidelity of a generation on an axis = human-consensus label (majority of the 3
annotators) matches the hidden target for that axis. Overall fidelity = all four
axes match (strict) and per-axis fidelity (lenient) are both reported.

Headline: the CONGRUENCE GAP = mean fidelity(congruent) - mean fidelity(incongruent),
with a bootstrap 95% CI and a permutation p-value. Broken down by model and axis.

Inputs:
  data/steerability/fidelity/fidelity_sheet_{A,B,C}.csv   (judged)
  data/steerability/fidelity/fidelity_key.csv             (hidden targets)
Outputs (data/steerability/fidelity/):
  fidelity_per_generation.csv   per-gen consensus label + match flags
  steerability_report.txt       printed summary (also to stdout)
"""

import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

AXES = ["intent", "primary_driver", "value_orientation", "affect"]
NP = "No-Persona"
rng = np.random.default_rng(0)


def truthy(x):
    return str(x).strip().lower() in {"1", "1.0", "true", "yes", "y", "x"}


def consensus(labels):
    labels = [l for l in labels if isinstance(l, str) and l.strip()]
    if not labels:
        return np.nan
    lab, n = Counter(labels).most_common(1)[0]
    return lab if n >= 2 else np.nan   # majority of 3; else tie -> undecided


def bootstrap_gap(cong, incong, n=10000):
    diffs = []
    for _ in range(n):
        a = rng.choice(cong, len(cong), replace=True).mean()
        b = rng.choice(incong, len(incong), replace=True).mean()
        diffs.append(a - b)
    return np.percentile(diffs, [2.5, 97.5])


def perm_pvalue(cong, incong, n=10000):
    obs = cong.mean() - incong.mean()
    pool = np.concatenate([cong, incong])
    k = len(cong)
    count = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[:k].mean() - pool[k:].mean()) >= abs(obs):
            count += 1
    return (count + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/steerability/fidelity")
    ap.add_argument("--sheets", nargs="+",
                    default=None, help="judged sheets; default A,B,C in --dir")
    args = ap.parse_args()
    d = Path(args.dir)
    sheets = args.sheets or [str(d / f"fidelity_sheet_{x}.csv") for x in ["A", "B", "C"]]

    key = pd.read_csv(d / "fidelity_key.csv").set_index("gen_id")
    raters = [pd.read_csv(s).set_index("gen_id") for s in sheets]

    ids = key.index
    rows = []
    for gid in ids:
        rec = {"gen_id": gid, "model": key.at[gid, "model"],
               "congruence": key.at[gid, "congruence"], "persona_id": key.at[gid, "persona_id"]}
        # per-axis consensus + match to target
        all_match = True
        for ax in AXES:
            labs = []
            for r in raters:
                if gid in r.index:
                    if "is_no_persona" in r.columns and truthy(r.at[gid, "is_no_persona"]):
                        labs.append(NP)
                    else:
                        v = r.at[gid, ax] if ax in r.columns else np.nan
                        labs.append(str(v).strip() if isinstance(v, str) else np.nan)
            cons = consensus(labs)
            tgt = key.at[gid, f"target_{ax}"]
            match = (isinstance(cons, str) and cons == tgt)
            rec[f"{ax}_consensus"] = cons
            rec[f"{ax}_target"] = tgt
            rec[f"{ax}_match"] = int(match)
            all_match = all_match and match
        rec["all_axes_match"] = int(all_match)
        rows.append(rec)

    fid = pd.DataFrame(rows)
    fid.to_csv(d / "fidelity_per_generation.csv", index=False)

    lines = []
    def out(s=""): lines.append(s); print(s)

    out("=" * 66)
    out("STEERABILITY FIDELITY  (human consensus matches target)")
    out("=" * 66)
    out(f"\nOverall strict fidelity (all 4 axes): {fid.all_axes_match.mean():.1%}")
    out("\nPer-axis fidelity:")
    for ax in AXES:
        out(f"  {ax:20s} {fid[f'{ax}_match'].mean():.1%}")

    out("\n" + "-" * 66)
    out("CONGRUENCE GAP  (primary result)")
    out("-" * 66)
    # use value_orientation axis as the primary congruence signal + strict overall
    for label, col in [("Value-Orientation axis", "value_orientation_match"),
                       ("all-axes (strict)", "all_axes_match")]:
        cong = fid.loc[fid.congruence == "congruent", col].values.astype(float)
        inc = fid.loc[fid.congruence == "incongruent", col].values.astype(float)
        gap = cong.mean() - inc.mean()
        lo, hi = bootstrap_gap(cong, inc)
        p = perm_pvalue(cong.copy(), inc.copy())
        out(f"\n[{label}]")
        out(f"  congruent   fidelity: {cong.mean():.1%}  (n={len(cong)})")
        out(f"  incongruent fidelity: {inc.mean():.1%}  (n={len(inc)})")
        out(f"  GAP = {gap:+.1%}   95% CI [{lo:+.1%}, {hi:+.1%}]   p={p:.4f}")

    out("\n" + "-" * 66)
    out("BY MODEL (all-axes strict fidelity, congruent vs incongruent)")
    out("-" * 66)
    for m, g in fid.groupby("model"):
        c = g.loc[g.congruence == "congruent", "all_axes_match"].mean()
        i = g.loc[g.congruence == "incongruent", "all_axes_match"].mean()
        out(f"  {m:10s} congruent {c:.1%} | incongruent {i:.1%} | gap {c-i:+.1%}")

    out("\n" + "-" * 66)
    out("BY PERSONA (all-axes strict fidelity)")
    out("-" * 66)
    for pid, g in fid.groupby("persona_id"):
        out(f"  {pid}: {g.all_axes_match.mean():.1%}  ({g.congruence.iloc[0]})")

    (d / "steerability_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {d}/fidelity_per_generation.csv , steerability_report.txt")


if __name__ == "__main__":
    main()
