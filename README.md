# PluralBench-NP

PluralBench-NP is a pluralistic benchmark focused on Steerable Pluralism in the Nepali language. Data was collected from various YouTube channels, capturing comments and public discourse around migration, brain drain, identity, and socio-economic pressure in Nepal.

The project is designed to support annotation, prompt development, and downstream evaluation of language models on Nepali cultural and value-oriented tasks.

## What is included

- Raw and filtered CSV datasets for multiple sources
- Prompt configuration for annotation workflows
- Preprocessing notebooks for comment filtering and dataset preparation

## Repository layout

```text
PluralBench-NP/
├── data/
│   ├── raw/
│   │   ├── IDS/
│   │   ├── Thaha Research/
│   │   └── The Nepali Comment/
│   └── filtered/
│       ├── IDS/
│       ├── Thaha Research/
│       └── The Nepali Comment/
├── prompts/
│   ├── annotation_prompt.md
│   ├── eval_prompt.md
│   └── llm_judge_prompt.md
├── src/
│   └── preprocessing/
│       └── filter_comments.ipynb
└── README.md
```

## Data overview

The `data/raw/` directory contains the original CSV exports, while `data/filtered/` contains cleaned or filtered versions of the same datasets.

Current sources include:

- `IDS`
- `Thaha Research`
- `The Nepali Comment`

## Prompt configuration

The annotation prompt used for NepPlural taxonomy classification lives in `prompts/annotation_prompt.md`. It defines the instructions for labeling comments as:

- `Intent`
- `Primary_Driver`
- `Value_Orientation`
- `Affect`

## Annotation goal

The taxonomy is intended to capture the underlying persona and motivation in a comment, rather than judging whether the comment is correct, polite, or offensive by Western moderation standards.

Examples of valid perspectives include:

- frustration with corruption or political instability
- pressure to migrate for work or family reasons
- praise or criticism tied to migration and social reality in Nepal

## Getting started

1. Explore the filtered datasets in `data/filtered/`.
2. Review the annotation prompt in `prompts/annotation_prompt.md`.
3. Use the preprocessing notebook in `src/preprocessing/filter_comments.ipynb` if you need to reproduce filtering or prepare new comment batches.

## Suggested workflow

1. Load a raw CSV from `data/raw/`.
2. Filter or normalize the comments.
3. Apply the NepPlural taxonomy prompt.
4. Store the cleaned output in `data/filtered/`.

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
