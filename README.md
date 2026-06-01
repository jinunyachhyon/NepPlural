# PluralBench-NP

PluralBench-NP is a Nepali-language benchmark and dataset for pluralistic value classification. It focuses on comments and public discourse around migration, brain drain, identity, and socio-economic pressure in Nepal.

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
├── src/
│   ├── preprocessing/
│   │   └── filter_comments.ipynb
│   └── prompts/
│       └── prompt_config.yaml
└── README.md
```

## Data overview

The `data/raw/` directory contains the original CSV exports, while `data/filtered/` contains cleaned or filtered versions of the same datasets.

Current sources include:

- `IDS`
- `Thaha Research`
- `The Nepali Comment`

## Prompt configuration

The annotation prompt used for NepPlural taxonomy classification lives in `src/prompts/prompt_config.yaml`. It defines the instructions for labeling comments as:

- `Validity`
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
2. Review the annotation prompt in `src/prompts/prompt_config.yaml`.
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

## License

See [LICENSE](LICENSE) for the project license.
