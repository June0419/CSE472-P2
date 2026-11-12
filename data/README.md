# Data

Keep original ARC-Challenge splits in `original/`, generated training data in
`protected/`, and the matched-budget random-edit control in `random_edits/`.
Generated data is ignored by Git. Preserve original example IDs, choice labels,
and answer keys. Only modify training question and choice text; keep evaluation
data clean. Record dataset revision, edit budget, seed, and generation settings.
