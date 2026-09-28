# Complete archive and restoration

The source inventory captured **6,162 files**, totaling **3,294,239,638 bytes**, including 1,060 notebooks (100 checkpoints). Original directory names, spaces, spelling, and checkpoint paths are retained so references can be traced directly. The original experiment folder is not modified.

## Portable Git contents

Each notebook has three representations:

1. `archive/notebooks/<original-path>.ipynb` preserves its cells, source, metadata, and execution counts. Saved outputs are moved into the output store.
2. `archive/python/<original-path>.py` contains all original code in cell order, including comments and plot code. Markdown is retained as comments. An import guard prevents running these experiments by importing them; use `src/gateattn` for reusable imports. Exported experiment code keeps historical paths and behavior.
3. `archive/output_index/<original-path>.json` maps each output-bearing cell to content-addressed compressed chunks in `archive/output_objects/`. Chunks retain full stream output, errors, HTML, images, and other MIME data, without truncation. The original JSON output is restored by concatenation and decompression.

Chunks are bounded at 4 MiB uncompressed and deduplicated by SHA-256. This preserves large visualization notebooks without putting files above GitHub's ordinary per-file limit into Git. Reconstruction is semantically exact JSON; byte-exact notebook JSON, including original formatting, remains in the local originals directory.

`archive/files/` contains all 47 original text logs. The manifest records every file's size, original SHA-256, destination, notebook reconstruction hash, and Python export path. `notebook_catalog.json` records model calls, literal configuration assignments, function names, output counts, and saved errors. These settings describe notebook source at capture time; saved outputs may originate from earlier interactive states.

## Local complete copy

`archive/originals/` contains the exact original bytes of every notebook. `local_assets/` contains the exact original bytes of every non-notebook file: PDF, images, COCO archives and annotations, and text logs. These two directories are Git-ignored to avoid duplicating notebook outputs and distributing large external assets in the source repository.

Nothing in the source inventory is omitted from the **local** archive. A Git clone contains all code, notebook outputs, logs, tabulated results, documentation, and the asset manifest; it does not contain the Git-ignored raw dataset/PDF copies. Obtain datasets from their original providers and use the manifest to verify an archival asset bundle if one is shared separately.

## Restore a notebook with all outputs

```bash
python tools/restore_notebook.py \
  'visualization/heatmap_red_yellow_visualization-original.ipynb' \
  --output runs/restored/heatmap_original.ipynb \
  --extract-media runs/restored/heatmap_original_images
```

The restored notebook includes the original saved outputs without rerunning inference. `--extract-media` writes embedded PNG, JPEG, SVG, and PDF outputs as individual files. It does not execute notebook code.

An original plot may be stored in a notebook whose name is not explicitly about visualization. Every notebook is preserved, including sweep generators, scratch notebooks, alternatives, and checkpoints. The [full notebook index](NOTEBOOK_INDEX.md) links to each source and Python export.

## Verify completeness

```bash
# After cloning: verify output chunks, notebook reconstruction hashes,
# Python AST equivalence to original cell code, and portable log hashes.
python tools/verify_archive.py

# On the original server, also verify the byte-exact local archive
# and the untouched parent source folder.
python tools/verify_archive.py --local --original-source ..
```

The archive creation tool is retained as `tools/archive_experiments.py`. It refuses to overwrite an existing manifest. For a future independent snapshot, create a new repository destination and pass `--source` and `--repo`; do not archive this repository into itself or replace the provenance of this release.
