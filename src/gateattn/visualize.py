"""Export entropy, patch selection, and per-block spatial attention figures."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from .evaluate import add_config_arguments, build_model, config_from_args, environment, hash_file, prepare, weight_digest


def save_figure(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.svg', '.png'):
        fig.savefig(stem.with_suffix(suffix), bbox_inches='tight', dpi=300)
    plt.close(fig)


def overlay_mask(image, mask, opacity=0.8):
    size = int(np.sqrt(len(mask)))
    if size * size != len(mask) or image.shape[0] % size or image.shape[1] % size:
        raise ValueError('Mask must describe a square grid aligned to the image.')
    keep = np.repeat(np.repeat(np.asarray(mask).reshape(size, size), image.shape[0] // size, axis=0), image.shape[1] // size, axis=1)
    output = image.astype(float).copy()
    output[~keep] = output[~keep] * (1 - opacity) + 255 * opacity
    return np.clip(output, 0, 255).astype(np.uint8)


def spatial_attention(snapshot, patch_count):
    """Scatter current tokens to original image positions; pruned positions are NaN."""
    grid = np.full(patch_count, np.nan)
    grid[snapshot.patch_indices.numpy()] = snapshot.scores.numpy()
    side = int(np.sqrt(patch_count))
    return grid.reshape(side, side)


def plot_entropy_panels(image, entropy, mask, output, caption=''):
    side = int(np.sqrt(len(entropy)))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    axes[0].imshow(image); axes[0].set_title('Input')
    heat = axes[1].imshow(entropy.reshape(side, side), cmap='magma', interpolation='nearest')
    axes[1].set_title('Patch entropy (bits)')
    fig.colorbar(heat, ax=axes[1], shrink=0.7)
    axes[2].imshow(mask.reshape(side, side), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[2].set_title(f'Retained patches: {int(mask.sum())}/{len(mask)}')
    axes[3].imshow(overlay_mask(image, mask)); axes[3].set_title('Pruned patches faded')
    for ax in axes:
        ax.set_axis_off()
    if caption:
        fig.suptitle(caption, fontsize=10)
    save_figure(fig, Path(output) / 'entropy_and_patch_selection')


def plot_attention_panels(image, snapshots, patch_count, output, prefix, caption=''):
    cols = 4
    rows = (len(snapshots) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows), squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap('autumn').copy()
    cmap.set_bad('white')
    arrays = {}
    for ax, snap in zip(axes.flat, snapshots):
        values = spatial_attention(snap, patch_count)
        arrays[f'block_{snap.block}'] = values
        ax.imshow(values, cmap=cmap, interpolation='nearest')
        ax.set_title(f'Block {snap.block}: {len(snap.patch_indices)} patches', fontsize=9)
        ax.set_axis_off()
    for ax in axes.flat[len(snapshots):]:
        ax.set_axis_off()
    fig.suptitle(f'{prefix.replace("_", " ").title()} — CLS attention (per-panel scale)\n{caption}', fontsize=11)
    save_figure(fig, Path(output) / prefix)
    np.savez_compressed(Path(output) / (prefix + '.npz'), **arrays)
    # Separate overlay figures retain exact spatial coordinates after every ATS step.
    for snap in snapshots:
        values = spatial_attention(snap, patch_count)
        expanded = np.repeat(np.repeat(values, image.shape[0] // values.shape[0], axis=0), image.shape[1] // values.shape[1], axis=1)
        mask = np.zeros(patch_count, dtype=bool)
        mask[snap.retained_indices.numpy()] = True
        fig, axes = plt.subplots(1, 2, figsize=(5, 2.6), constrained_layout=True)
        axes[0].imshow(image); axes[0].imshow(expanded, cmap=cmap, alpha=0.55)
        axes[0].set_title(f'Block {snap.block} attention')
        axes[1].imshow(overlay_mask(image, mask)); axes[1].set_title('Retained after block')
        for ax in axes:
            ax.set_axis_off()
        if caption:
            fig.suptitle(caption, fontsize=8)
        save_figure(fig, Path(output) / f'{prefix}_block_{snap.block:02d}')
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-unpruned', action='store_true')
    args = parser.parse_args()
    config = config_from_args(args)
    if args.output.exists():
        raise FileExistsError('Choose a new visualization output directory.')
    model = build_model(config, args.device, capture_attention=True)
    tensor, mask, entropy, image = prepare(args.image, model, config, np.random.default_rng(config.seed))
    args.output.mkdir(parents=True)
    caption = 'Random initialization — software demonstration only' if config.random_init else config.model
    with torch.inference_mode():
        logits = model(tensor.to(args.device), mask)
        snapshots = model.snapshots
        token_counts = model.token_counts
        plot_entropy_panels(image, entropy, mask, args.output, caption)
        plot_attention_panels(image, snapshots, len(mask), args.output, 'pruned_attention', caption)
        unpruned_top1 = None
        if args.compare_unpruned:
            model.rates = {}
            unpruned = model(tensor.to(args.device))
            unpruned_top1 = int(unpruned.argmax())
            plot_attention_panels(image, model.snapshots, len(mask), args.output, 'unpruned_attention', caption)
    np.savez_compressed(args.output / 'patch_data.npz', entropy_bits=entropy, keep_mask=mask, rgb=image)
    meta = {'config': asdict(config), 'environment': environment(), 'image_sha256': hash_file(args.image),
            'weights_sha256': weight_digest(model), 'pruned_top1_index': int(logits.argmax()),
            'unpruned_top1_index': unpruned_top1, 'block_input_tokens': token_counts,
            'attention_scale': 'each panel scales independently; inspect NPZ for quantitative comparison',
            'snapshots': [{'block': s.block, 'patch_indices': s.patch_indices.tolist(),
                           'retained_indices': s.retained_indices.tolist()} for s in snapshots]}
    (args.output / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Figures and numeric arrays saved to {args.output}')


if __name__ == '__main__':
    main()
