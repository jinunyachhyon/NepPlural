# PluralBench-NP

PluralBench-NP is a pluralistic benchmark focused on Steerable Pluralism in the Nepali language. Data was collected from various YouTube channels, capturing comments and public discourse around migration, brain drain, identity, and socio-economic pressure in Nepal.

The project is designed to support annotation, prompt development, and downstream evaluation of language models on Nepali cultural and value-oriented tasks.

## What is included

- Raw and filtered CSV datasets for three YouTube sources
- Multi-model LLM annotations and majority-voted labels under the NepPlural taxonomy
- Human-reviewed final annotations plus a 20% sample for quality checks
- LLM-as-a-Judge verification of the sampled labels, with aggregate statistics
- Prompts for annotation, evaluation, and judging
- Notebooks covering preprocessing, annotation, validation, and NLU fine-tuning
- Fine-tuning results for five transformer baselines

## Repository layout

```text
PluralBench-NP/
├── data/
│   ├── raw/                         # original CSV exports per source
│   ├── filtered/                    # cleaned/filtered comments
│   ├── annotations/
│   │   ├── label/                   # per-model labels (GPT-5.2, Gemini 3.5 flash, Sonet 4.6)
│   │   ├── Majority_Voting/         # labels merged by majority vote across models
│   │   ├── Final_Annotations/       # human-reviewed, cleared labels
│   │   ├── Final_Annotations_sample_20/  # 20% sample of the final annotations
│   │   └── AnnotationGuidelines.pdf
│   └── LLM_Judge_Verification/      # judge verdicts on the 20% sample
│       └── verification_stats.json  # per-file and total wrong/correct rates
├── prompts/
│   ├── annotation_prompt.md
│   ├── eval_prompt.md
│   └── llm_judge_prompt.md
├── src/
│   ├── preprocessing/
│   │   └── filter_comments.ipynb
│   ├── annotation/
│   │   ├── majority_voting.ipynb
│   │   └── resolve_human_reviews.ipynb
│   ├── validation/
│   │   ├── check_annotations.ipynb
│   │   └── sample_20_percent.ipynb
│   └── training/
│       ├── finetune_nlu.ipynb
│       └── Results/                 # metrics + test predictions per model
└── README.md
```

Each source subfolder (`IDS`, `Thaha Research`, `The Nepali Comment`) is mirrored
across the `data/` stages, so a single dataset can be traced from raw export
through filtering, labelling, voting, human review, and judge verification.

## Data overview

The data pipeline moves through several stages:

1. `data/raw/` — original CSV exports from each source.
2. `data/filtered/` — cleaned and filtered comments (see `src/preprocessing/`).
3. `data/annotations/label/` — independent labels from each LLM annotator.
4. `data/annotations/Majority_Voting/` — labels combined by majority vote.
5. `data/annotations/Final_Annotations/` — human-reviewed, cleared labels.
6. `data/annotations/Final_Annotations_sample_20/` — a 20% sample for QA.
7. `data/LLM_Judge_Verification/` — judge verdicts on that sample.

Current sources include:

- `IDS`
- `Thaha Research`
- `The Nepali Comment`

## Annotation and verification

Labels are produced by three LLM annotators (GPT-5.2, Gemini 3.5 flash, and
Sonet 4.6), merged by majority vote, then reviewed by humans to form the final
annotations. A 20% sample is then re-checked by an LLM-as-a-Judge using
`prompts/llm_judge_prompt.md`: a row is marked wrong if any of its four persona
labels is judged not defensible. Per-file and overall wrong/correct rates are
recorded in `data/LLM_Judge_Verification/verification_stats.json`.

## NLU fine-tuning

`src/training/finetune_nlu.ipynb` fine-tunes transformer models to predict the
four NepPlural dimensions. Metrics (per-dimension macro-F1 and accuracy plus a
mean macro-F1) and test predictions for each model are stored under
`src/training/Results/`. Baselines include multilingual BERT, XLM-RoBERTa, two
IRIIS Research Nepali models, and NepBERTa.

## Prompt configuration

The annotation prompt used for NepPlural taxonomy classification lives in
`prompts/annotation_prompt.md`. It defines the instructions for labeling comments
across `Intent`, `Primary_Driver`, `Value_Orientation`, and `Affect`. The
evaluation and judging prompts live in `prompts/eval_prompt.md` and
`prompts/llm_judge_prompt.md`.

## Annotation goal

The taxonomy is intended to capture the underlying persona and motivation in a comment, rather than judging whether the comment is correct, polite, or offensive by Western moderation standards.

Examples of valid perspectives include:

- frustration with corruption or political instability
- pressure to migrate for work or family reasons
- praise or criticism tied to migration and social reality in Nepal

## Getting started

1. Explore the datasets in `data/`, following the stages in the data overview above.
2. Review the prompts in `prompts/`.
3. Use the notebooks in `src/` to reproduce any stage of the pipeline.
4. Inspect fine-tuning results in `src/training/Results/`.

## Suggested workflow

1. Load a raw CSV from `data/raw/`.
2. Filter or normalize the comments with `src/preprocessing/filter_comments.ipynb`.
3. Apply the NepPlural taxonomy prompt to produce per-model labels.
4. Merge labels by majority vote (`src/annotation/majority_voting.ipynb`) and resolve human reviews (`src/annotation/resolve_human_reviews.ipynb`).
5. Sample and check annotations (`src/validation/`), then run LLM-as-a-Judge verification.
6. Fine-tune and evaluate NLU baselines with `src/training/finetune_nlu.ipynb`.

## Notes

- Comments may be written in Devanagari Nepali, Romanized Nepali, or English.
- Political profanity and emotional frustration can still be valid perspectives when they relate to Nepali socio-economic discourse.
- The project is structured for annotation and analysis, not moderation.

## Label taxonomy

This project uses the NepPlural multi-dimensional taxonomy. Each comment is
labeled across the following categories (values listed with short descriptions):

- **Intent** (stance on migration): the commenter's personal stance or plan.
	- `Pro-Migration`: Wants to leave, plans to migrate, or advises leaving.
	- `Anti-Migration`: Prefers staying, urges building locally, or returned home.
	- `Trapped/Regretful`: Wants to migrate but cannot, or migrated and regrets it.
	- `Neutral/Observation`: Describes the situation without stating personal intent.

- **Primary_Driver**: the root cause motivating the commenter's view.
	- `Economic Necessity`: Money, unemployment, poverty, or survival needs.
	- `Family Obligation`: Duty to family, loans, remittances, or parental pressure.
	- `Systemic/Political Anger`: Corruption, politicians, inequality, or failing institutions.
	- `Patriotism/Love`: Attachment to culture, land, or national identity.

- **Value_Orientation**: who/what the commenter prioritizes.
	- `Collectivist-Family`: Sacrifices for family welfare or reputation.
	- `Collectivist-Nation`: Prioritizes national or societal benefit.
	- `Individualist-Self`: Prioritizing own career, growth, peace of mind, or personal wealth.

- **Affect**: the emotional tone of the comment.
	- `Despairing/Sad`: Helplessness, sadness, or resignation.
	- `Angry/Frustrated`: Aggression, profanity, sarcasm, intense irritation at a target.
	- `Hopeful/Motivated`: Optimism, resilience, or a call to action.
	- `Pragmatic`: Cold, calculated, emotionless statement of facts/plans.

## License

See [LICENSE](LICENSE) for the project license.
