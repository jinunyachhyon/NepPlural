# NepPlural annotation — allowed values

Label every comment on all four axes. If a comment expresses no migration
persona (off-topic / pure praise), tick `is_no_persona` = 1 and leave the axes blank.
Work independently — do not discuss individual comments with the other annotator.

**intent**: Pro-Migration | Anti-Migration | Trapped/Regretful | Neutral/Observation
**primary_driver**: Economic Necessity | Family Obligation | Systemic/Political Anger | Patriotism/Love
**value_orientation**: Collectivist-Family | Collectivist-Nation | Individualist-Self
**affect**: Despairing/Sad | Angry/Frustrated | Hopeful/Motivated | Pragmatic

`is_no_persona`: 1 if off-topic/no persona, else 0
`notes`: optional free text for hard cases

Final labels are set by majority vote across annotators (2-of-3).
Do not edit, sort, or reorder the `comment_id` column — it is the merge key.