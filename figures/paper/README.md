# JSA main research graphs

These PNGs are the unchanged saved outputs of the [supplied Colab notebook](https://colab.research.google.com/drive/1Of57FctGySefxJt7GaN-1cbma_IRbtX1?usp=sharing), matched visually and by plotted series to Figures 7–9 in `JSA_main.pdf` (pages 8–9). They are not plots generated from the paper's tables.

| Paper figure | Original Colab cell index (zero-based) | Saved output |
| --- | --- | --- |
| 7, Small/Tiny panel | 16 | [figure7_small_tiny.png](figure7_small_tiny.png) |
| 7, Small/Tiny baseline legend | 17 | [figure7_small_tiny_legend.png](figure7_small_tiny_legend.png) |
| 7, Base/Large panel | 14 | [figure7_base_large.png](figure7_base_large.png) |
| 7, Base/Large baseline legend | 15 | [figure7_base_large_legend.png](figure7_base_large_legend.png) |
| 8, GOPs bars and accuracy lines | 31 | [figure8_accuracy_gops.png](figure8_accuracy_gops.png) |
| 9, six-component latency breakdown | 25 | [figure9_latency.png](figure9_latency.png) |

Only these six outputs are included: four components of Figure 7, plus Figures 8 and 9. Exploratory variants, Swin bar charts, plots made from tables, and synthetic demonstrations are excluded from this collection.

## Relationship to the final paper layout

The Colab outputs are source graph assets, not pixel-identical exports of the final typeset figures:

- Figure 7 combines two panels and rearranges the baseline legends in the paper. The paper adds a shaded pruning region and uses the x-axis label “EDPG Token Pruning (%)”; the saved notebook panels say “Token Pruning (%)” and have no shading.
- Figure 8 uses cell 31, with GOPs savings on the left axis, accuracy on the right, and blue/orange/green/yellow lines matching the paper. Earlier cells use different axes, chart types, or line colors.
- Figure 9 uses cell 25 with the six components and legend labels 0, 10, and 20. The paper adds numerical annotations over the bars. Earlier notebook variants include a separate pruning-overhead category absent from the paper.

The original notebook outputs are preserved without adding or changing these presentation details. Both the latency and accuracy plots retain the original nonlinear axis scaling.

## Re-extraction and provenance

Run from the repository root:

```bash
python tools/extract_paper_figures.py
```

The tool decodes the outputs in [the selected source notebook](../../notebooks/jsa_paper_graphs.ipynb), verifies all six output hashes, and then writes the PNG files. It never executes notebook code. The notebook retains only the selected original cells; their plotting code depends on the original interactive state and is retained for provenance, not presented as a standalone executable notebook.

[manifest.json](manifest.json) records the downloaded full notebook hash, paper hash, stable Colab cell IDs, original zero-based cell/output positions, and exported PNG hashes. You can also supply a downloaded copy of the full original notebook with `--notebook /path/to/notebook.ipynb`; only the approved outputs with matching hashes will be extracted.

The original experiment archive is preserved separately. Generated run images remain local and are ignored by Git. No new SVG or PDF versions have been fabricated from these PNG outputs.
