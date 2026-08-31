# Steerability generation — how to run

108 generations = 3 models x 6 personas x 6 questions.

For each row in generation_sheet.csv:
1. Open the model's playground (AI Studio for Gemini, OpenAI playground for GPT, Anthropic console for Claude).
2. Paste `system_instruction` into the SYSTEM / instructions field.
3. Paste `question` as the user message.
4. Generate. Paste the output into `response`.
5. Record `model_version` (exact, e.g. gemini-2.5-flash), `temperature` (read the actual default off the panel — don't write 'default'), and `date`.

Keep temperature the SAME across all rows for one model so the comparison is fair. Do not edit the target_* columns — those are the ground truth for scoring, and the annotators will NOT see them.

Tip: generate model-by-model (all Gemini rows, then all GPT, then all Claude) so you only set up each playground once.
