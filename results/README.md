# Results and provenance

## `paper/`

Nine CSVs represent all eight numbered paper tables; Table 4 is split into parts a and b. `figure9_latency.csv` contains the latency components stated in the paper. Every row includes a provenance label and source location. `tools/build_paper_tables.py` deterministically recreates these files.

Reported accuracy, GOP values, savings percentages, parameter counts, hardware figures, and rounding are transcribed as printed. Derived `accuracy_change_pp` columns are differences from the corresponding table's baseline. These files do not imply a successful rerun of the refactored code.

## `extracted/`

- `notebook_accuracy_history.csv`: all 50,738 matching accuracy messages in saved notebook streams, including checkpoints and partial runs, with notebook/cell/output provenance.
- `notebook_last_observations.csv`: the last saved observation from each of 498 notebooks containing matching messages, along with maximum observed image count and saved-error status.
- `text_log_accuracy_history.csv`: 55,595 matching lines from original text logs, with line numbers and conservative segment boundaries when processed-image counts reset.

The catalog contains 551 notebooks with saved error outputs, many being user interrupts. Their intermediate measurements remain part of the archive. A saved-error flag does not determine whether every earlier cell result is invalid; conversely, the absence of an error does not prove a complete run. Completion is deliberately unverified for recovered observations.

`model_from_source` and pruning fields are inferred from the notebook's stored code, not validated execution history. Some notebook names differ from model calls; use the catalog and source to investigate rather than identifying experiments by filename alone. Text logs do not contain enough metadata to reliably identify every appended run's settings, so settings are not guessed from filenames.

For a publication curve from recovered runs, explicitly identify the intended notebook/output, checkpoint, evaluation image count, preprocessing, and matching baseline. Keep the chosen source list with the curve. The repository does not average duplicates or choose whichever run has the highest accuracy.

## New measurements

Use fresh directories under `runs/`. Each completed run has metadata, prediction records, and a summary from `gateattn-evaluate`. Preserve these files together. Random-initialized CPU smoke tests are marked and are not research accuracy results.
