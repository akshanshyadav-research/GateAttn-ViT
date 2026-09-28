# Method and implementation conventions

## Entropy-driven patch gate

For each patch, the implementation counts pixel intensities and computes `H = -sum(p * log2(p))`, excluding zero-probability bins. The usual input is a 224×224 image with 16×16 patches: 196 spatial tokens and one CLS token.

`entropy_mode="gray"` uses OpenCV RGB-to-grayscale conversion. `rgb` pools the intensity values of all three channels into one histogram, matching the RGB notebook convention; it is not joint RGB-vector entropy. Inputs must be uint8.

The paper's Algorithm 1 prints a blue grayscale coefficient of `0.144`. The supplied notebook code uses `cv2.cvtColor`; the reusable implementation follows that executable source convention rather than changing it to the printed coefficient.

`entropy_policy` controls cutoff behavior:

| Policy | Retention and ties |
| --- | --- |
| `legacy` (default) | `max(1, floor(N * (1-drop)))` tokens selected by NumPy `argsort`, as in the original function |
| `stable` | Same exact count, with lower spatial indices winning ties |
| `threshold` | All tokens at or above the kth-highest entropy; tied values can retain extra tokens |

At least one spatial patch is retained. Entropy pruning occurs after patch projection and positional embedding, before the first encoder block. Full-image tokenization cost therefore remains.

## Attention-based token selection

The model uses the existing timm transformer blocks and pretrained parameters. Attention hooks capture the mean CLS attention over heads. ATS computes retention from that attention and removes selected spatial tokens **after the block's attention, residual, and MLP computations**, as in the notebook `CustomBlock`.

- Block indices are **zero-based**. `[3, 6, 9]` is applied after those three blocks.
- `attention_policy="quantile"` includes CLS in `torch.quantile`, matches the notebooks' threshold convention, and always preserves CLS. Ties may produce variable token counts.
- `attention_policy="topk"` ranks only spatial tokens with stable ties and keeps an exact floor-based count.
- A scalar `attention_drop` applies at every ATS block; JSON configurations can instead contain a list of rates aligned with `attention_blocks`.
- The wrapper accepts **one image at a time**. Variable-length batches are not silently reshaped.
- Retained tokens stay in their original order. The wrapper records original patch IDs for all attention visualizations.
- The wrapper is for fixed-resolution timm ViTs with one CLS token. Swin, register-token variants, and distilled DeiT with two prefix tokens are not silently treated as standard ViTs.

The attention hook disables fused attention where probabilities are required. This permits transparent inspection but changes runtime and may produce small numerical differences from a fused kernel. Reported software timings include this instrumentation.

## Preprocessing and numerical behavior

The `legacy` path follows the notebooks: nearest-neighbor RGB resize to the model resolution, conversion to `[0,1]`, no ImageNet normalization; entropy uses an independently bilinear-resized grayscale image. The `timm` path uses the selected checkpoint's evaluation transform and computes entropy on its transformed spatial grid. These paths are separate experimental protocols and must not be mixed in a curve.

The original custom transformer used `nn.LayerNorm` with its default epsilon of `1e-5`. Configurations preserve that value. Set `layer_norm_eps` to `null` in JSON to retain the timm checkpoint architecture's epsilon. The new runner calls `eval()` and `torch.inference_mode()` explicitly, uses sorted image enumeration, and records seeds. This is a controlled evaluation implementation; it does not assert bitwise replay of every historical notebook run.

Current timm names may resolve to different pretrained defaults than the historical environment. For repeatable research, use the exact checkpoint file or explicit timm pretrained tag, record the weight hash, and compare matched baselines.

## Fixed-point simulation

`quantization="notebook-fx16"` rounds and saturates Linear/Conv2d weights and biases and the input tensor, using signed `word_bits=16`, `fractional_bits=8`. Other operations and intermediate activations remain floating point. This follows the scope of the provided `QuantizedVisionTransformer` wrapper; it is **not an integer kernel implementation or full FPGA FX16 arithmetic simulator**. `torch.round` uses its own documented rounding convention; boundary behavior should be checked against QPyTorch for exact legacy replay.

The name `notebook-fx16` identifies that simulation scope. FP32 evaluation uses `quantization="none"`.

## Baselines

`random` uses a seeded NumPy generator. `center-square` retains the original notebook's central `floor(sqrt(k)) × floor(sqrt(k))` square; actual patch drop may be larger than the requested percentage. `center` retains exactly k closest-to-center patches, breaking distance ties by spatial index. The two center variants are labeled separately. The runner logs actual counts.

Similarity-related code and logs are preserved in the archive, including commented experimental alternatives. No new similarity method is presented as a verified reproduction of a published result.

## Compute and timing

`complexity.py` estimates MACs for patch projection, QKV and output projection, attention products, both MLP projections, and the classifier. With embedding dimension D, MLP width M, and block input length N, encoder cost is `4ND² + 2N²D + 2NDM` MACs. Norms, biases, softmax, entropy, sorting, and memory movement are excluded. Multiplying by two uses the FLOPs convention of two operations per MAC.

The paper's GOP values are kept unchanged in the published-data CSVs. Analytical MAC estimates are stored under a different field name and are not presented as the paper's GOP measurements. Single-image model-forward time is synchronized on CUDA, includes no warmup, and is not a hardware benchmark.
