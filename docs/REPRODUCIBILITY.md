# Reproduction guide

## Installation and validation

Use Python 3.10 and the versions in `requirements.txt`. The actual local validation environment is recorded in `validation_environment.json`. CUDA wheels are platform-specific; select a matching PyTorch installation before installing this project. The reusable implementation does not require QPyTorch, transformers, scikit-image, or papermill. Those dependencies belong to historical notebooks and are available through the optional `legacy` extra; their historical versions are unknown.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python tools/verify_archive.py --local --git-tracked
```

Tests use CPU and small random-initialized models; they do not download weights. They cover exact entropy-mask parity with the archived notebook function, known entropy distributions, baseline counts, fixed-point saturation, ATS tie behavior, CLS retention, original spatial indices, unpruned timm prediction parity, input validation, class mapping, and evaluation provenance.

GitHub Actions retrieves LFS objects, repeats CPU tests, verifies the complete original-file archive, checks command entry points, and renders published-data figures. It does not run the large historical notebook collection or claim ImageNet reproduction.

## Match baselines first

Use the baseline and pruned config for the same architecture, preprocessing, epsilon, checkpoint, and quantization:

```bash
gateattn-evaluate --config configs/vit_base_baseline.json \
  --data /path/to/val --class-map /path/to/class_map.json \
  --device cuda:0 --output runs/base-unpruned

gateattn-evaluate --config configs/vit_base_combined.json \
  --data /path/to/val --class-map /path/to/class_map.json \
  --device cuda:0 --output runs/base-combined
```

The supplied configs request notebook-style fixed-point simulation and legacy preprocessing. To compare full-precision checkpoint-standard behavior, use `--quantization none --preprocessing timm` and set `layer_norm_eps` to `null` in the JSON for both runs. That is a separate protocol from historical notebook measurements.

## Sweeps

`tools/sweep.py` writes numbered JSON configurations by default. `--run` executes each in sequence, with its own output directory:

```bash
python tools/sweep.py --suite combined \
  --models vit_tiny_patch16_224 vit_small_patch16_224 vit_base_patch16_224 \
  --data /path/to/val --class-map /path/to/class_map.json \
  --device cuda:0 --output runs/combined --run
```

The combined suite uses the nine Table 2 configurations, including baseline. Entropy and placement suites accept arbitrary rate lists. A 1%–50% sweep can be generated with a Bash sequence:

```bash
python tools/sweep.py --suite entropy \
  --rates $(python -c 'print(" ".join(str(i/100) for i in range(51)))') \
  --output runs/entropy-0-to-50
```

For random dropping, pass several seeds with `--seeds 0 1 2 3 4`. The configuration and seed are saved per run. Results from differing seeds should be summarized with a stated aggregation method; they are not pooled automatically.

## Output interpretation

- `metadata.json`: resolved config, actual weight hash, package versions, data/class-map identity, running/complete/failed status, and metric kind.
- `predictions.jsonl`: image path and hash, target, top-five predictions, exact patch and block token counts, analytical MAC estimate, and synchronized forward duration.
- `summary.json`: completed-run accuracy, number of images, mean analytical MACs, timing, and scope.

A failed run retains its metadata and completed prediction lines, but is not given a completed summary. Existing output directories are refused. There is no implicit resume; use a fresh output path.

Input processing and image hashing are excluded from the per-image forward timer. The overall wall time includes them. There is no warmup, and hooks/materialized attention affect the implementation's latency. Use a dedicated benchmark protocol before making throughput comparisons.

## Results available without inference

```bash
python tools/extract_results.py
python tools/build_paper_tables.py
python tools/extract_paper_figures.py
```

The extraction tool derives summaries from the already archived notebook observations and parses all text logs. Saved results may be partial, duplicated, or from a previous interactive state. The script preserves their source and saved-error indicators instead of inferring which one is the definitive published run.

See `results/README.md` and `docs/PAPER_TO_CODE.md` for provenance and coverage. Full ImageNet evaluation and FPGA deployment have not been performed as part of this repository preparation.

## Plot completed new evaluations

```bash
python tools/plot_runs.py \
  --runs runs/base-unpruned runs/base-combined \
  --output runs/base-comparison
```

The plot uses analytical GMAC estimates and accuracy from completed software runs. A companion CSV records each selected run and the checksums of its metadata and summary. Select matched protocols and datasets; this utility does not infer a baseline or connect unlike measurements into a curve. Random-initialized runs are excluded unless explicitly requested and are then labeled as smoke tests.
