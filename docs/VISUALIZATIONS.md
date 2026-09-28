# Visualization guide

## Reusable figures

`gateattn-visualize` generates:

- A four-panel input / entropy / retention / faded-patch figure.
- A full set of mean CLS attention maps across every encoder block.
- Individual attention overlays and retained-patch illustrations for every block.
- Optional corresponding unpruned figures from the same model state.
- PDF, SVG, and 300-dpi PNG exports, plus NPZ arrays and JSON metadata.

```bash
gateattn-visualize --config configs/vit_base_combined.json \
  --image /path/to/image.jpg --compare-unpruned \
  --device cuda:0 --output runs/visualization
```

The `pruned_attention.npz` and `unpruned_attention.npz` files store each block's attention on the original image grid. Pruned positions are NaN, not reassigned to adjacent pixels. For ATS blocks, attention is measured before selection and the paired retention panel shows the tokens left after selection. Each panel is scaled independently for visibility; use the raw arrays for numerical comparisons.

`metadata.json` records image and weight hashes, the full config, selected patch indices, per-block counts, and predicted class indices. Random-initialized examples are explicitly labeled in figures.

## Original visualization families

| Original directory / notebook | Preserved material |
| --- | --- |
| `visualization/heatmap_red_yellow_visualization*.ipynb` | Original red/yellow attention, entropy and image plots, including complex-image variants |
| `vision_pipeline/notebook_entropy_heatmap_visualization.ipynb` | Patch entropy maps and input views |
| `vision_pipeline/entropy+attnetion_collab_visualization.ipynb` | Entropy/attention combinations and highlighted attention |
| `vision_pipeline/patch_drop_visualization.ipynb` | Dropped-patch visualization code and outputs |
| `vision_pipeline/heatmap_red_yellow_visualization.ipynb` | Additional heatmap implementation |
| `centre_crop/`, `vit_base/` | Baseline and dataset exploration figures embedded in experiment notebooks |
| All other notebooks and checkpoints | Every plot cell and saved MIME output retained |

Python exports preserve **all** original cells, including exploratory or commented plotting code. No visualization notebook was selected as a replacement for the others.

To recover an original notebook and its embedded figures without executing it:

```bash
python tools/restore_notebook.py \
  'vision_pipeline/notebook_entropy_heatmap_visualization.ipynb' \
  --output runs/original-entropy-heatmaps.ipynb \
  --extract-media runs/original-entropy-figures
```

The full [notebook index](NOTEBOOK_INDEX.md) includes every variant. Original image files are tracked in `local_assets/` through Git LFS and inventoried by SHA-256. Run `git lfs pull` after cloning to retrieve those inputs for rerunning original visualizations. Saved notebook outputs can also be restored from the compressed output store without rerunning inference.

## Paper graphs from Colab

The published graph collection contains only the original saved Colab outputs corresponding to Figures 7–9 in `JSA_main.pdf`. See [the figure mapping](../figures/paper/README.md) for exact source cell IDs, hashes, and differences between the notebook outputs and the final paper layout.

```bash
python tools/extract_paper_figures.py
```

This decodes the selected saved PNG outputs without executing the notebook or redrawing any data. The selected source cells and outputs are preserved in `notebooks/jsa_paper_graphs.ipynb`.

`gateattn-plot` remains an optional local analysis tool: it creates new plots from table CSVs, not the paper's original figures. Its outputs, synthetic demos, and generated run images are not part of the published figure collection. The original experiment archive remains intact.
