# NepPlural: A Human-Anchored Benchmark for Persona Annotation in Nepali Migration Discourse

NepPlural is a benchmark dataset and evaluation suite for modelling competing
socio-cultural personas in **Nepali** public discourse on youth migration and
brain drain. It pairs a corpus of code-mixed (Devanagari / Romanized Nepali /
English) YouTube comments with a four-axis persona taxonomy, a **human-adjudicated
gold standard**, a five-encoder baseline suite, and a human-judged generative
**steerability** evaluation.

> **Anonymized for double-blind review.** This repository contains no author
> names or identifying information. Please do not attempt to de-anonymize.

## The taxonomy

Every comment is annotated on four orthogonal axes, plus an explicit No-Persona class:

- **Intent** (stance): Pro-Migration, Anti-Migration, Trapped/Regretful, Neutral/Observation
- **Primary Driver**: Economic Necessity, Family Obligation, Systemic/Political Anger, Patriotism/Love
- **Value Orientation**: Collectivist-Family, Collectivist-Nation, Individualist-Self
- **Affect**: Despairing/Sad, Angry/Frustrated, Hopeful/Motivated, Pragmatic
- **No-Persona / Off-Topic**: the comment expresses no migration persona

## Dataset

- **Sources**: comment sections of three Nepali YouTube channels (IDS, Thaha
  Research, The Nepali Comment) covering migration podcasts, interviews, and
  public commentary.
- **Corpus**: 2,148 filtered code-mixed comments; a deterministic, auditable
  router (no LLM) separates a substantive pool (1,894) from off-topic comments (254).
- **Human gold**: a stratified sample of **300 comments** independently labelled by
  **three native Nepali speakers** on all four axes, combined by majority vote with
  three-way ties adjudicated.

## Key findings

- **Humans agree; LLMs do not.** Native-speaker inter-annotator agreement is
  moderate (Fleiss' kappa: Intent 0.56, Primary Driver 0.58, Value Orientation
  0.55, Affect 0.53), while three frontier LLMs on the same comments reach only
  near-chance agreement (kappa 0.13-0.22). An independent LLM-as-a-Judge pass flags
  35% of automatic labels, concentrated in off-topic comments. The taxonomy is
  well-defined for people who understand the discourse; the models are the
  limiting factor.
- **LLMs over-attribute personas.** Native speakers judge 116/300 (38.7%) of
  substantive-looking comments to carry no migration persona; the non-abstaining
  LLM pipeline assigns a persona to all of them.
- **Human ceiling ~0.80** mean macro-F1 (annotator vs adjudicated gold).
- **Encoder benchmark** (5 models x 5 seeds, tested on human gold): multilingual
  encoders lead (XLM-R and mBERT ~0.34 mean macro-F1) over Nepali-specific models
  (~0.20), about 42% of the human ceiling. (Multilingual vs Nepali-specific is
  confounded with pretraining scale.)
- **Steerability** (108 human-judged generations, three frontier models): models
  adopt single target axes at 59-77% fidelity but compose full personas correctly
  only 33% of the time. No congruence gap is detected between Western-aligned and
  Nepali-collectivist personas at this scale (gap ~0%, p=1.0).

## Repository structure

```
data/
  raw/ filtered/            source + cleaned comments (PII removed)
  annotations/              LLM labels, majority vote, judge verification
  LLM_Judge_Verification/   20% judge-verification results + stats
  pools/                    substantive_pool / no_persona; sample/ + step3/ gold
  splits/                   train / val / test (test = 300 human gold)
  steerability/             generation set, personas, fidelity sheets, results
src/
  preprocessing/            filtering, No-Persona routing, pooling, sampling
  analysis/                 agreement scoring, gold finalization, splits, ceiling,
                            steerability set + scoring
  training/                 multi-task encoder trainer + aggregation
prompts/                    annotation / LLM-judge / steerability-eval prompts
```

## Reproducing the pipeline

```bash
pip install -r requirements.txt

# 1. preprocess + pool
python src/preprocessing/prepare_pools.py
python src/preprocessing/join_llm_labels.py
python src/preprocessing/stratified_sample.py

# 2. human annotation happens here (see data/pools/sample/)
python src/analysis/score_annotations.py --sheets <A> <B> <C>
python src/analysis/finalize_gold.py

# 3. build splits + reference lines
python src/analysis/build_splits.py
python src/analysis/human_ceiling.py --sheets <A> <B> <C>

# 4. train + aggregate (GPU)
for m in <models>; do python src/training/train_multitask.py --model "$m"; done
python src/training/aggregate_results.py

# 5. steerability (manual generation between steps 1 and 2 below)
python src/analysis/build_steerability_set.py
python src/analysis/build_fidelity_sheets.py
python src/analysis/score_steerability.py
```

## Data statement and ethics

This dataset is derived from public YouTube comments on a politically sensitive
topic. Personal identifiers were removed; only anonymized comment text is
released. See `DATA_STATEMENT.md` for provenance, anonymization, re-identification
considerations, and intended use. Labels describe the persona a comment expresses,
not the truth of any claim, and must not be used to profile or target individuals
or communities.

## License

Released under the Creative Commons Attribution 4.0 International (CC BY 4.0)
license, for both the dataset and the code. See `LICENSE`.

## Citation

Citation information will be added after review.
