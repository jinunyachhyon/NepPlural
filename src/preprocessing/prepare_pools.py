#!/usr/bin/env python3
"""
prepare_pools.py

One pass from the per-source CSVs to two clean pools:

  substantive_pool.csv : comments with a potential migration persona  -> HUMAN ANNOTATION
  no_persona.csv       : off-topic / praise / content-empty comments   -> No-Persona/Off-Topic label
                         (kept in the released dataset, NOT annotated)

What it does:
  1. Reads every CSV under --in_glob (default: data/filtered/**/*.csv), across all
     source subfolders.
  2. Normalizes the comment column (some files use `Comment`, one uses `Text`).
  3. DROPS personally identifiable columns (Author, etc.) — nothing that identifies a
     commenter leaves this stage.
  4. Derives source_channel (parent folder) and source_file, and a stable comment_id.
  5. Routes each comment deterministically (no LLM) into substantive vs no_persona.

Defaults to `data/filtered/` because that stage is already deduplicated and
spam/empty-filtered. Point --in_glob at data/raw/ only if you also re-run
dedup + PII scrubbing yourself.
"""

import re
import html
import glob
import os
import hashlib
import argparse
from pathlib import Path
import pandas as pd

# --------------------------------------------------------------------------- regex
WORD_RE     = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)
URL_RE      = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE  = re.compile(r"@\w+")
HASHTAG_RE  = re.compile(r"#\w+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
EMOJI_RE    = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF\U00002B00-\U00002BFF\u2764\u2665\uFE0F\u200D]",
    flags=re.UNICODE,
)

# --------------------------------------------------------------------------- column aliasing / PII
COMMENT_ALIASES = ["comment", "text"]              # normalize these to `comment`
PII_COLS        = ["author", "channel", "user", "username", "handle", "parent id"]  # dropped

# --------------------------------------------------------------------------- lexicons
SUBSTANTIVE = {
    # migration / abroad + destinations Nepali migrants actually go to
    "bidesh","videsh","bidesi","videshi","foreign","abroad","migrate","migration",
    "immigrant","gulf","dubai","qatar","katar","malaysia","malesia","korea","japan",
    "america","amrika","usa","uk","europe","yurop","australia","canada","visa","pr",
    "passport","manpower","remittance","dollar","permit","jhitko","kaam","work",
    "student","study","dv","lahure","lahur","portugal","poland","romania","croatia",
    "uae","saudi","arab","israel","germany","france","spain","italy","greece","cyprus",
    "malta","kuwait","bahrain","oman","singapore","china","india","russia","ireland",
    "denmark","norway","sweden","finland","netherlands","belgium","newzealand",
    "pugera","puger","bidesbata","kamaune","kamai","bhabishya","future",
    # political / systemic
    "sarkar","sarakar","neta","netaharu","corruption","bhrastachar","bhrasta",
    "byabastha","system","politician","government","desh","देश","सरकार","नेता",
    "भ्रष्टाचार","राजनीति","rajniti","lutera","luteko","lutta","luttantra",
    # economic
    "paisa","पैसा","rojgar","रोजगार","berojgar","बेरोजगार","unemploy","garibi",
    "गरिबी","poverty","salary","tankha","job","mehanat","gaas","gari khana",
    # family / obligation
    "parivar","परिवार","aama","आमा","buba","बुबा","chhoro","chhori","ghar","घर",
    "family","parents","aafanta","remit",
    # migration verbs / stance (roman)
    "janu","gaye","gayo","jane","farkanu","farkera","basnu","basne","chodne",
}

PRAISE = {
    "nice","good","great","amazing","awesome","excellent","wonderful","beautiful",
    "best","lovely","love","superb","fantastic","brilliant","perfect","informative",
    "wow","waw","respect","salute","proud","congrats","congratulations","thanks",
    "thank","thankyou","weldone","welldone","done","bravo","keep","going","keepitup",
    "inspiring","inspirational","motivational","fabulous","fire","goat","w","wid",
    "vid","video","content","subscribe","subscribed","first","fan","hi","hello",
    "ramro","राम्रो","mitho","sundar","सुन्दर","badhai","बधाई","dhanyabad","धन्यवाद",
    "subhakamana","शुभकामना","sahi","सहि","सही","thik","wah","वाह","jindabad",
    "zindabad","maya","mayaa",
}
GENERIC_SINGLE = {"sad","true","real","yes","no","ok","okay","fact","facts","hmm",
                  "lol","haha","omg","great video","nice video","good video"}


def clean(t):        return HTML_TAG_RE.sub(" ", html.unescape("" if t is None else str(t))).strip()
def word_tokens(t):  return WORD_RE.findall(EMOJI_RE.sub(" ", HASHTAG_RE.sub(" ", MENTION_RE.sub(" ", URL_RE.sub(" ", t)))))

def substantive_hit(t):
    low = t.lower()
    for kw in SUBSTANTIVE:
        if kw in low:
            return kw
    return None

def route(text):
    t = clean(text)
    toks = word_tokens(t)
    n = len(toks)
    if n == 0:
        return "no_persona", "content_empty", None
    kw = substantive_hit(t)
    if kw is not None:
        return "substantive", "substantive_keyword", kw
    low = [w.lower() for w in toks]
    if " ".join(low) in GENERIC_SINGLE:
        return "no_persona", "generic_reaction", None
    hits = sum(1 for w in low if w in PRAISE)
    if n <= 8 and hits >= 1:
        return "no_persona", f"praise_short(hits={hits})", None
    if n and hits / n >= 0.5 and hits >= 2:
        return "no_persona", f"praise_dominant(frac={hits/n:.2f})", None
    return "substantive", "kept_default", None


def find_comment_col(cols):
    lower = {c.lower(): c for c in cols}
    for a in COMMENT_ALIASES:
        if a in lower:
            return lower[a]
    return None


def load_all(in_glob):
    rows = []
    for f in sorted(glob.glob(in_glob, recursive=True)):
        df = pd.read_csv(f)
        ccol = find_comment_col(df.columns)
        if ccol is None:
            print(f"  !! skipping {f}: no comment/text column"); continue
        rel = os.path.relpath(f)
        parts = Path(rel).parts
        channel = parts[-2] if len(parts) >= 2 else "unknown"      # parent folder = source channel
        sfile = Path(rel).stem
        keep_meta = [c for c in df.columns
                     if c.lower() in ("script", "language", "word_count", "emoji_count")]
        for _, r in df.iterrows():
            rec = {"comment": r[ccol], "source_channel": channel, "source_file": sfile}
            for c in keep_meta:
                rec[c.lower()] = r[c]
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", default="data/filtered/**/*.csv")
    ap.add_argument("--out_dir", default="data/pools")
    args = ap.parse_args()

    df = load_all(args.in_glob)
    df["comment"] = df["comment"].astype(str)
    df["comment_id"] = df["comment"].apply(
        lambda s: hashlib.sha1(s.strip().lower().encode("utf-8")).hexdigest()[:12])

    r = df["comment"].apply(route).apply(pd.Series)
    r.columns = ["bucket", "route_reason", "substantive_hit"]
    df = pd.concat([df, r], axis=1)

    # order columns; PII already never loaded
    front = ["comment_id", "comment", "source_channel", "source_file",
             "script", "language", "word_count", "emoji_count",
             "bucket", "route_reason", "substantive_hit"]
    df = df[[c for c in front if c in df.columns]]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    sub = df[df.bucket == "substantive"].drop(columns=["bucket"])
    nop = df[df.bucket == "no_persona"].drop(columns=["bucket"])
    sub.to_csv(f"{args.out_dir}/substantive_pool.csv", index=False)
    nop.to_csv(f"{args.out_dir}/no_persona.csv", index=False)

    print(f"TOTAL loaded   {len(df)}")
    print(f"substantive    {len(sub):4d}  ({100*len(sub)/len(df):.1f}%)  -> data/pools/substantive_pool.csv (annotate)")
    print(f"no_persona     {len(nop):4d}  ({100*len(nop)/len(df):.1f}%)  -> data/pools/no_persona.csv (No-Persona label)")
    print("\nper source_channel (substantive only):")
    print(sub.groupby("source_channel").size().to_string())
    print("\nroute_reason breakdown:")
    print(df.groupby("route_reason").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
