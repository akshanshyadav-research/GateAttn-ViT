# Paper-to-code and evidence map

Paper: **GateAttn-ViT: Entropy-gated, attention-guided token pruning for resource-efficient Vision Transformer acceleration on FPGAs**, Akshansh Yadav and Palash Das, Journal of Systems Architecture 177 (2026), 103836. DOI: `10.1016/j.sysarc.2026.103836`.

The supplied `JSA_main.pdf` is tracked through Git LFS under `local_assets/`; its exact checksum is in `archive/manifest.json`. The table transcription script is `tools/build_paper_tables.py`.

| Paper component | Available source evidence | Reusable entry point / data | Scope |
| --- | --- | --- | --- |
| EDPG / Algorithm 1 | Entropy functions throughout original ViT/DeiT notebooks | `entropy.py`, `evaluate.py` | Executable software implementation and original code preserved |
| ATS / combined pruning | `ATS_layers 3,6,9/`, `pipeline_with _attention/` | `model.py`, combined configs and sweep | Executable software implementation; zero-based block convention documented |
| FX16 numerical experiments / Table 1 | QPyTorch wrappers in notebooks | `quantization.py`, `table1_models.csv` | Notebook-style weight/input simulation; not full fixed-point hardware arithmetic |
| Table 2 | Combined-pruning notebook family and saved logs | `table2_pruning.csv`, combined sweep | Published table transcribed; no asserted one-to-one linkage to a validated final notebook run |
| Table 3 | Paper only for hardware measurements | `table3_pruning_hardware.csv` | HLS/RTL and synthesis artifacts not supplied |
| Figure 7 / entropy sweeps | Model-size folders, `notebooks1…7`, `output1…7` | Entropy sweep; extracted observation CSVs | Saved observations preserved; final canonical curve not inferred from duplicate or interrupted runs |
| Figure 8 / pruning placement | ATS and entropy notebook variants | Placement sweep | New controlled runner; original code and outputs retained |
| Table 4 / Swin | No Swin implementation found in supplied notebooks | `table4a_swin_independent.csv`, `table4b_swin_combined.csv` | Published data only; not covered by the single-CLS ViT wrapper |
| Figure 9 / FPGA latency | Paper text and figure | `figure9_latency.csv`, figure generator | Published component times, not software timings |
| Table 5 / FPGA comparison | Paper only for hardware results | `table5_hardware.csv` | Published measurements transcribed; no bitstream, board run, or power report supplied |
| Table 6 / prior-work comparison | Paper comparison | `table6_comparison.csv` | Values as reported in this paper; external methods not reimplemented |
| Table 7 / CIFAR-100 and ObjectNet | No matching implementations found in supplied notebooks | `table7_datasets.csv` | Published data only; dataset-specific weights, mappings, and code not supplied |
| Table 8 / simple baselines | `centre_crop/`, random-drop notebooks under `vit_tiny/` | Center/random APIs, baseline sweep, `table8_baselines.csv` | Original square-center convention and exact-count alternative distinguished |
| Attention / entropy illustrations | `visualization/`, `vision_pipeline/`, other notebook outputs | `visualize.py`, full original Python exports and output restoration | Original plot code and saved images preserved; new spatially indexed visualization API |
| Additional exploration | `vit_base/` ImageNet-A and COCO-related assets | Complete notebook / asset archive | Preserved as exploratory material; no new dataset-specific reproduction claim |

## Published-number conventions

All paper CSV rows explicitly state `transcribed_from_supplied_paper_not_rerun`. Reported savings are preserved rather than recomputed from rounded GOP numbers. For example, the ViT-Tiny Table 2 `20/15/15/15` row lists 0.81 GOPs and 38.85% savings; those published values are retained as-is. Table 8 uses a ViT-Small baseline of 78.11%, while Table 2 uses 78.01%; neither is silently replaced with the other.

The listed Figure 9 components sum to 50.947 ms at 20% and 70.435 ms at baseline. At 10%, the printed components sum to 63.559 ms. The latency CSV keeps component and total fields separately to make such checks possible.

Recorded notebook source, saved outputs, and published values are three evidence layers. In an interactive notebook they need not have been generated from the same final source state. This repository preserves all three and avoids assigning full-reproduction status without an end-to-end rerun with identified weights and data.
