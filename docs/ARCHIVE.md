# Complete archive and restoration

The source inventory captured **6,162 files**, totaling **3,294,239,638 bytes**, including 1,060 notebooks (100 checkpoints). Original directory names, spaces, spelling, and checkpoint paths are retained so references can be traced directly. The original experiment folder is not modified.

## Portable Git contents

Each notebook has three representations:

1. `archive/notebooks/<original-path>.ipynb` preserves its cells, source, metadata, and execution counts. Saved outputs are moved into the output store.
2. `archive/python/<original-path>.py` contains all original code in cell order, including comments and plot code. Markdown is retained as comments. An import guard prevents running these experiments by importing them; use `src/gateattn` for reusable imports. Exported experiment code keeps historical paths and behavior.
3. `archive/output_index/<original-path>.json` maps each output-bearing cell to content-addressed compressed chunks in `archive/output_objects/`. Chunks retain full stream output, errors, HTML, images, and other MIME data, without truncation. The original JSON output is restored by concatenation and decompression.

Chunks are bounded at 4 MiB uncompressed and deduplicated by SHA-256. This preserves large visualization notebooks without putting files above GitHub's ordinary per-file limit into Git. Reconstruction is semantically exact JSON; byte-exact notebook JSON, including original formatting, remains in the local originals directory.

`archive/files/` contains all 47 original text logs. The manifest records every file's size, original SHA-256, destination, notebook reconstruction hash, and Python export path. `notebook_catalog.json` records model calls, literal configuration assignments, function names, output counts, and saved errors. These settings describe notebook source at capture time; saved outputs may originate from earlier interactive states.

## Complete original files in Git LFS

`archive/originals/` contains the exact original bytes of every notebook. `local_assets/` contains the exact original bytes of every non-notebook file: PDF, images, COCO archives and annotations, and text logs. Both directories are tracked through Git LFS, preserving exact bytes while keeping large files out of ordinary Git blobs. The historical directory name `local_assets` is retained for compatibility; these assets are now part of the repository.

Nothing in the source inventory is omitted from Git. A complete clone with Git LFS objects includes every original file as well as all code, notebook outputs, logs, tabulated results, documentation, and generated run artifacts. Install Git LFS and run `git lfs pull` after cloning. An ordinary `git archive` or hosting-service ZIP may contain LFS pointers instead of these large files; use an LFS-enabled clone for the complete experiment.

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

# After git lfs pull, verify all original bytes and their tracked LFS identities.
python tools/verify_archive.py --local --git-tracked
git lfs fsck

# On the original server, also verify the untouched parent source folder.
python tools/verify_archive.py --local --git-tracked --original-source ..
```

The archive creation tool is retained as `tools/archive_experiments.py`. It refuses to overwrite an existing manifest. For a future independent snapshot, create a new repository destination and pass `--source` and `--repo`; do not archive this repository into itself or replace the provenance of this release.
