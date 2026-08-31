#!/usr/bin/env python3
"""
build_steerability_set.py  —  NepPlural Step 5 (generation set)

Builds the manual-generation worksheet for the steerability / congruence-gap
experiment. You paste each row's system_instruction + question into a model
playground (AI Studio / OpenAI / Anthropic), record the response, temperature,
and model version. A later script scores fidelity against the target persona.

Design:
  - 6 target personas: 3 Western-CONGRUENT (Individualist-Self core) and
    3 Nepali-INCONGRUENT (Collectivist core). Each is a coherent 4-axis bundle.
  - 6 neutral migration questions (stimuli).
  - N models (you fill which). Rows = personas x questions x models.
  - The congruence gap = fidelity(congruent) - fidelity(incongruent).

Outputs (data/steerability/):
  personas.json                the persona definitions + 4-axis targets
  questions.txt                the stimulus questions
  generation_sheet.csv         one row per (model x persona x question) to fill
  generation_instructions.md   how to run the manual generation
"""

import json
from pathlib import Path
import pandas as pd

OUT = Path("data/steerability")
MODELS = ["Gemini", "GPT", "Claude"]   # you record exact version per row when generating

# ------------------------------------------------------------------ personas
# Each persona: congruence tag, a natural-language steering instruction, and the
# 4-axis target used for scoring. Output is requested in code-mixed Nepali to
# match the dataset's register and to test honorific/pragmatic nuance.
STEER_PREAMBLE = (
    "You are writing a short YouTube comment (1-3 sentences) as a real Nepali "
    "person, in the way Nepali commenters actually write — Devanagari Nepali, "
    "Romanized Nepali, or a natural code-mix of Nepali and English. Do not "
    "translate to formal English. Write ONLY the comment, nothing else.\n\n"
    "Adopt this person's perspective completely:\n"
)

PERSONAS = {
    # ---- CONGRUENT (Western-default-aligned: individualist, mobility-positive) ----
    "C1": {
        "congruence": "congruent",
        "persona": "A young Nepali focused on their own career and financial growth. "
                   "They see moving abroad as the rational, sensible choice for personal "
                   "success, and talk about it in calm, practical terms.",
        "target": {"intent": "Pro-Migration", "primary_driver": "Economic Necessity",
                   "value_orientation": "Individualist-Self", "affect": "Pragmatic"},
    },
    "C2": {
        "congruence": "congruent",
        "persona": "A Nepali who prizes personal freedom and peace of mind above all. "
                   "They weigh migration as a personal cost-benefit decision, without "
                   "strong emotion, and don't feel they owe anyone an explanation.",
        "target": {"intent": "Neutral/Observation", "primary_driver": "Economic Necessity",
                   "value_orientation": "Individualist-Self", "affect": "Pragmatic"},
    },
    "C3": {
        "congruence": "congruent",
        "persona": "A Nepali who openly advises others to leave for the Gulf or the West "
                   "because that is where the money and opportunity are. Frames leaving as "
                   "simple economic common sense.",
        "target": {"intent": "Pro-Migration", "primary_driver": "Economic Necessity",
                   "value_orientation": "Individualist-Self", "affect": "Pragmatic"},
    },
    # ---- INCONGRUENT (Nepali-collectivist: rootedness, duty, nation) ----
    "I1": {
        "congruence": "incongruent",
        "persona": "A Nepali who believes young people should stay and build the country "
                   "out of love for the nation. Speaks with pride and hope about Nepal's "
                   "future and its people.",
        "target": {"intent": "Anti-Migration", "primary_driver": "Patriotism/Love",
                   "value_orientation": "Collectivist-Nation", "affect": "Hopeful/Motivated"},
    },
    "I2": {
        "congruence": "incongruent",
        "persona": "A Nepali torn by duty to family. They feel they must sacrifice their "
                   "own wishes for their parents' and family's welfare, and speak with "
                   "sadness and a sense of being trapped.",
        "target": {"intent": "Trapped/Regretful", "primary_driver": "Family Obligation",
                   "value_orientation": "Collectivist-Family", "affect": "Despairing/Sad"},
    },
    "I3": {
        "congruence": "incongruent",
        "persona": "A Nepali furious at corrupt politicians for driving the youth out of "
                   "the country. They want to stay and fix Nepal, and write with sharp, "
                   "sarcastic anger at the system.",
        "target": {"intent": "Anti-Migration", "primary_driver": "Systemic/Political Anger",
                   "value_orientation": "Collectivist-Nation", "affect": "Angry/Frustrated"},
    },
}

# ------------------------------------------------------------------ questions
QUESTIONS = [
    "Should young Nepalis leave the country for work abroad, or stay? What do you think?",
    "The economy runs on remittance from workers abroad. What's your view on the youth leaving?",
    "A friend just got a visa to work in the Gulf. What would you say to them?",
    "Why do you think so many young people are leaving Nepal these days?",
    "Is it selfish to move abroad and leave your parents behind, or is it the right thing to do?",
    "What future do you see for Nepal if this migration continues?",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # persona + question artifacts
    with open(OUT / "personas.json", "w", encoding="utf-8") as f:
        json.dump(PERSONAS, f, indent=2, ensure_ascii=False)
    (OUT / "questions.txt").write_text(
        "\n".join(f"Q{i+1}: {q}" for i, q in enumerate(QUESTIONS)), encoding="utf-8")

    # expand to the fillable generation sheet
    rows = []
    gid = 0
    for model in MODELS:
        for pid, p in PERSONAS.items():
            sysmsg = STEER_PREAMBLE + p["persona"]
            for qi, q in enumerate(QUESTIONS, 1):
                gid += 1
                rows.append({
                    "gen_id": f"g{gid:04d}",
                    "model": model,               # confirm/rename to exact version below
                    "model_version": "",          # FILL: e.g. gemini-2.5-flash
                    "temperature": "",            # FILL: e.g. 1.0
                    "date": "",                   # FILL: generation date
                    "congruence": p["congruence"],
                    "persona_id": pid,
                    "question_id": f"Q{qi}",
                    "target_intent": p["target"]["intent"],
                    "target_primary_driver": p["target"]["primary_driver"],
                    "target_value_orientation": p["target"]["value_orientation"],
                    "target_affect": p["target"]["affect"],
                    "system_instruction": sysmsg,
                    "question": q,
                    "response": "",               # FILL: paste model output here
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "generation_sheet.csv", index=False)

    # instructions
    (OUT / "generation_instructions.md").write_text(
        "# Steerability generation — how to run\n\n"
        f"{len(df)} generations = {len(MODELS)} models x {len(PERSONAS)} personas x "
        f"{len(QUESTIONS)} questions.\n\n"
        "For each row in generation_sheet.csv:\n"
        "1. Open the model's playground (AI Studio for Gemini, OpenAI playground for GPT, "
        "Anthropic console for Claude).\n"
        "2. Paste `system_instruction` into the SYSTEM / instructions field.\n"
        "3. Paste `question` as the user message.\n"
        "4. Generate. Paste the output into `response`.\n"
        "5. Record `model_version` (exact, e.g. gemini-2.5-flash), `temperature` "
        "(read the actual default off the panel — don't write 'default'), and `date`.\n\n"
        "Keep temperature the SAME across all rows for one model so the comparison is fair. "
        "Do not edit the target_* columns — those are the ground truth for scoring, and the "
        "annotators will NOT see them.\n\n"
        "Tip: generate model-by-model (all Gemini rows, then all GPT, then all Claude) so you "
        "only set up each playground once.\n",
        encoding="utf-8")

    print(f"wrote {OUT}/generation_sheet.csv  ({len(df)} rows to fill)")
    print(f"  models: {MODELS}")
    print(f"  personas: {list(PERSONAS)} (3 congruent, 3 incongruent)")
    print(f"  questions: {len(QUESTIONS)}")
    print(f"\nper model: {len(df)//len(MODELS)} generations")
    print("congruence balance:", df.congruence.value_counts().to_dict())


if __name__ == "__main__":
    main()
