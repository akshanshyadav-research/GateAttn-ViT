"""Render publication tables as PDF/SVG/300-dpi PNG figures without a GPU."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .visualize import save_figure


def read_csv(root, name):
    with (root / name).open() as stream:
        return list(csv.DictReader(stream))


def render_paper(data, output):
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    rows = read_csv(data, 'table2_pruning.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for model in ['ViT-Tiny','ViT-Small','ViT-Base']:
        selected = sorted([r for r in rows if r['model'] == model], key=lambda r: float(r['savings_pct_reported']))
        x = [float(r['savings_pct_reported']) for r in selected]
        axes[0].plot(x, [float(r['top1_pct']) for r in selected], 'o-', label=model)
        axes[1].plot(x, [-float(r['accuracy_change_pp']) for r in selected], 'o-', label=model)
    for ax in axes:
        ax.set_xlabel('Reported computational savings (%)'); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel('Top-1 accuracy (%)'); axes[1].set_ylabel('Accuracy loss (percentage points)')
    fig.suptitle('Table 2 — published results (transcribed)')
    save_figure(fig, output / 'table2_accuracy_compute')
    rows = read_csv(data, 'table8_baselines.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for model, ax in zip(['ViT-Tiny','ViT-Small'], axes):
        selected = [r for r in rows if r['model'] == model]
        for key, label in [('center_crop_top1_pct','Center crop'),('random_drop_top1_pct','Random drop'),('edpg_top1_pct','EDPG')]:
            ax.plot([float(r['drop_pct']) for r in selected], [float(r[key]) for r in selected], 'o-', label=label)
        ax.set(title=model, xlabel='Reported drop rate (%)', ylabel='Top-1 accuracy (%)'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Table 8 — published baseline comparison (transcribed)')
    save_figure(fig, output / 'table8_baselines')
    rows = read_csv(data, 'table7_datasets.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for key, label, ax in zip(['cifar100_top1_pct','objectnet_top1_pct'], ['CIFAR-100','ObjectNet'], axes):
        for model in ['ViT-Tiny','ViT-Small']:
            selected = [r for r in rows if r['model'] == model]
            ax.plot([float(r['drop_pct']) for r in selected], [float(r[key]) for r in selected], 'o-', label=model)
        ax.set(title=label, xlabel='Reported drop rate (%)', ylabel='Top-1 accuracy (%)'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Table 7 — published dataset results (transcribed)')
    save_figure(fig, output / 'table7_datasets')
    rows = read_csv(data, 'figure9_latency.csv')
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    bottoms = np.zeros(len(rows))
    for key, label in [('tokenization_ms','Tokenization'),('msa_ms','MSA'),('mlp_ms','MLP'),('layernorm_ms','LayerNorm'),('residual_ms','Residual'),('vit_io_ms','ViT I/O')]:
        vals = np.array([float(r[key]) for r in rows])
        ax.bar([r['uniform_pruning_pct'] + '%' for r in rows], vals, bottom=bottoms, label=label)
        bottoms += vals
    ax.set(xlabel='Uniform pruning at −1, 3, 6, 9', ylabel='Latency (ms)', title='Figure 9 — published FPGA latency components')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    save_figure(fig, output / 'figure9_latency')
    rows = [r for r in read_csv(data, 'table5_hardware.csv') if r['method'] == 'GateAttn-ViT' and r['frequency_mhz'] == '250']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for key, label, ax in zip(['fps','fps_per_w'],['Throughput (FPS)','Efficiency (FPS/W)'],axes):
        ax.bar([r['pruning_pct'] for r in rows], [float(r[key]) for r in rows], color='#357ca5')
        ax.set(ylabel=label, xlabel='Pruning percentages (−1/3/6/9)'); ax.tick_params(axis='x', rotation=20)
    fig.suptitle('Table 5 — published ZCU104 measurements at 250 MHz')
    save_figure(fig, output / 'table5_hardware')
    rows = read_csv(data, 'table4b_swin_combined.csv')
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for model in ['Swin-Tiny','Swin-Small','Swin-Base']:
        selected = [r for r in rows if r['model'] == model]
        ax.plot([r['drop_percent_at_minus1_1'] for r in selected], [float(r['top1_pct']) for r in selected], 'o-', label=model)
    ax.set(xlabel='Pruning percentages (−1/1)', ylabel='Top-1 accuracy (%)', title='Table 4b — published Swin results (transcribed)'); ax.legend(); ax.grid(alpha=.2)
    save_figure(fig, output / 'table4b_swin')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('results/paper'))
    p.add_argument('--output', type=Path, default=Path('figures/paper'))
    args = p.parse_args()
    render_paper(args.data, args.output)
    print(f'Paper-based figures written to {args.output}')


if __name__ == '__main__':
    main()
