# Contributing

Keep historical artifacts under `archive/` unchanged. Add new evaluations under a fresh output directory and include their resolved configuration, checkpoint identity, dataset/class-map identity, image count, preprocessing, quantization, seed, and actual token counts when sharing results.

Run `python -m unittest discover -s tests -v` for changes to the reusable package. If archive tooling changes, also run `python tools/verify_archive.py`; local maintainers can add `--local --original-source ..` to verify exact originals. Test numerical changes against an unpruned matched baseline and label changed protocols explicitly.

Paper-transcribed numbers belong in `results/paper/`; recovered historical observations belong in `results/extracted/`; new measurements should have their own provenance. Do not replace missing results with estimates.
