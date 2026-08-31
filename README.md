# NepPlural: A Human-Anchored Benchmark for Pluralistic Alignment in Nepali Migration Discourse

NepPlural is a benchmark dataset and evaluation suite for modelling competing
socio-cultural perspectives in **Nepali** public discourse on youth migration and
brain drain. It pairs a corpus of code-mixed (Devanagari / Romanized Nepali /
English) YouTube comments with a four-axis persona taxonomy, a **human-adjudicated
gold standard**, encoder baselines, and a generative **steerability** evaluation.

> This is the v2, human-anchored release. The earlier v1 pipeline relied on
> LLM-only labels; v2 adds independent native-speaker annotation, treats LLM
> labels as a compared signal rather than ground truth, and adds an explicit
> No-Persona / Off-Topic class. Both pipelines are retained for transparency.

## The taxonomy

Every comment is annotated on four orthogonal axes, plus a No-Persona flag:

- **Intent** (stance): Pro-Migration, Anti-Migration, Trapped/Regretful, Neutral/Observation
- **Primary Driver**: Economic Necessity, Family Obligation, Systemic/Political Anger, Patriotism/Love
- **Value Orientation**: Collectivist-Family, Collectivist-Nation, Individualist-Self
- **Affect**: Despairing/Sad, Angry/Frustrated, Hopeful/Motivated, Pragmatic
- **No-Persona / Off-Topic**: the comment expresses no migration persona

## Dataset

- **Sources**: comment sections of three Nepali YouTube channels — IDS, Thaha
  Research, and The Nepali Comment — covering migration podcasts, interviews, and
  public commentary. Collected [FILL: date range] from [FILL: N] videos.
- **Corpus**: 2,148 filtered code-mixed comments; a deterministic router separates
  a substantive pool (1,894) from off-topic comments (254).
- **Human gold**: a stratified sample of **300 comments** independently labelled by
  **three native Nepali speakers** on all four axes, with 3-way ties adjudicated.

### Key findings

- **Humans agree; LLMs do not.** Native-speaker inter-annotator agreement is
  moderate (Fleiss' kappa: intent 0.56, primary_driver 0.58, value_orientation
  0.55, affect 0.53); three frontier LLMs labelling the same comments reach only
  near-chance agreement (kappa 0.10-0.22). The taxonomy is well-defined for people
  who understand the discourse; the models are the limiting factor.
- **LLMs over-attribute personas.** Human annotators judged 116/300 (38.7%) of
  substantive-looking comments to carry no migration persona; the LLM pipeline,
  which never abstains, assigned a persona to all of them.
- **Human ceiling ~0.80** mean macro-F1 (individual annotator vs adjudicated gold).
- **Encoder benchmark** (5 models x 5 seeds, tested on human gold): multilingual
  encoders lead (XLM-R and mBERT ~0.34 mean macro-F1) over Nepali-specific models
  (~0.20) -- about 42% of the human ceiling. No-Persona is the hardest class
  (F1 0.20-0.28). Multilingual vs Nepali-specific is confounded with pretraining scale.
- **Steerability** (generative, congruence-gap experiment, human-judged over 108
  generations from three frontier models): models adopt single target axes at
  59-77% fidelity but compose all four axes correctly only 33% of the time. We do
  NOT detect a congruence gap between Western-congruent and Nepali-collectivist
  personas at this scale (gap ~0%, p=1.0); steering difficulty appears symmetric
  rather than directionally biased.

## Repository structure

```
data/
  raw/ filtered/            source + cleaned comments (PII removed)
  annotations/              v1 LLM labels, majority vote, judge verification
  pools/                    substantive_pool / no_persona; sample/ + step3/ gold
  splits/                   train/val/test (test = 300 human gold)
  steerability/             generation set, personas, fidelity sheets, results
src/
  preprocessing/            filter, routing, pooling, sampling
  analysis/                 agreement, gold finalize, splits, ceiling, steerability
  training/                 multi-task encoder trainer + aggregation
prompts/                    annotation / judge / eval prompts
```

## Reproducing the pipeline

```bash
pip install -r requirements.txt

# preprocess + pool
python src/preprocessing/prepare_pools.py
python src/preprocessing/join_llm_labels.py
python src/preprocessing/stratified_sample.py

# after human annotation (see data/pools/sample/)
python src/analysis/score_annotations.py --sheets <A> <B> <C>
python src/analysis/finalize_gold.py
python src/analysis/build_splits.py
python src/analysis/human_ceiling.py --sheets <A> <B> <C>

# train + aggregate (GPU)
for m in <models>; do python src/training/train_multitask.py --model "$m"; done
python src/training/aggregate_results.py

# steerability
python src/analysis/build_steerability_set.py
python src/analysis/build_fidelity_sheets.py
python src/analysis/score_steerability.py
```

## License

Released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**
license, for both the dataset and the code. See `LICENSE`.

