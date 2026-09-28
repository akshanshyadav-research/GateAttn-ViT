# Using GateAttn-ViT

## 1. Download the complete project

Install Git and Git LFS, then run:

```bash
git lfs install
git clone https://github.com/IITjodhpurbuddy/GateAttn-ViT.git
cd GateAttn-ViT
git lfs pull
```

The complete archive is several gigabytes because it includes the original notebooks, saved outputs, images, and datasets. Use Git LFS to retrieve the actual files; a source ZIP may contain pointers instead.

## 2. Install the software

Python 3.10 is the validated version. On Linux or macOS:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

On Windows, use `py -3.10 -m venv .venv` and activate with `.venv\Scripts\Activate.ps1` in PowerShell. The commands above install CPU PyTorch; use a compatible CUDA build of the same PyTorch/torchvision versions for GPU evaluation.

## 3. View or extract the paper graphs

Figures 7–9 are displayed in the [README](../README.md). View the saved notebook outputs in [`notebooks/jsa_paper_graphs.ipynb`](../notebooks/jsa_paper_graphs.ipynb), or extract their original PNG bytes:

```bash
python tools/extract_paper_figures.py
```

This command needs only Python's standard library. It checks the saved output hashes and does not execute the notebook, download model weights, or rerun experiments. [Graph provenance and paper-layout differences](../figures/paper/README.md).

## 4. Verify the installation and archive

```bash
python -m unittest discover -s tests -v
python tools/verify_archive.py --local --git-tracked
git lfs fsck
```

The archive verification checks all 6,162 original files and reconstructs all 1,060 original notebooks.

## 5. Evaluate your validation images

Prepare class-organized images and a JSON mapping to the checkpoint's class indices:

```text
/path/to/validation/
  n01440764/image_001.jpg
  n01443537/image_002.jpg
```

The full class mapping and ImageNet validation set must be supplied separately. `configs/class_map.example.json` illustrates the format for two classes; it is not a complete ImageNet mapping. See [data setup](DATA.md).

Run a baseline:

```bash
gateattn-evaluate \
  --config configs/vit_base_baseline.json \
  --data /path/to/validation \
  --class-map /path/to/imagenet_class_index.json \
  --device cpu \
  --output runs/my-baseline
```

Run combined EDPG/ATS pruning on the same data:

```bash
gateattn-evaluate \
  --config configs/vit_base_combined.json \
  --data /path/to/validation \
  --class-map /path/to/imagenet_class_index.json \
  --device cpu \
  --output runs/my-combined
```

Use `--device cuda:0` for a compatible GPU, or `--max-images 100` for a limited check. Pretrained model weights download on first use unless a compatible local `--checkpoint` is provided. Choose a fresh output directory for each run.

Each evaluation produces `summary.json`, `predictions.jsonl`, and `metadata.json`. These record accuracy, predictions, timing scope, configuration, and model/data provenance. Software timings are not FPGA latency measurements.

## 6. Visualize an image

```bash
gateattn-visualize \
  --config configs/vit_base_combined.json \
  --image /path/to/image.jpg \
  --compare-unpruned \
  --device cpu \
  --output runs/my-visualization
```

This creates entropy, patch-retention, and attention views for your image. Generated run graphs stay local; the repository's paper graph collection contains only the selected Colab outputs.

## 7. Explore the original experiments

Use the [notebook index](NOTEBOOK_INDEX.md) to find an experiment. Recover an original notebook and its saved output images without running it:

```bash
python tools/restore_notebook.py \
  'vision_pipeline/notebook_entropy_heatmap_visualization.ipynb' \
  --output runs/recovered-entropy.ipynb \
  --extract-media runs/recovered-entropy-images
```

The reusable runner has configurable paths; original notebooks retain their historical paths and environment assumptions. See [reproduction details](REPRODUCIBILITY.md), [method conventions](METHOD.md), and [paper-to-code coverage](PAPER_TO_CODE.md) before comparing new measurements with the paper.
