#!/usr/bin/env python3
"""Plot explicitly selected completed evaluations, retaining run provenance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gateattn.visualize import save_figure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-smoke-tests', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new output directory.')
    rows = []
    for run in args.runs:
        meta_path, summary_path = run/'metadata.json', run/'summary.json'
        meta, summary = json.loads(meta_path.read_text()), json.loads(summary_path.read_text())
        if meta['status'] != 'complete' or summary['status'] != 'complete':
            p.error(f'Incomplete run: {run}')
        if meta['metric_kind'] == 'random_initialization_smoke_test' and not args.include_smoke_tests:
            p.error('Random-initialized runs require --include-smoke-tests and are not research results.')
        config = meta['config']
        rows.append({'run': str(run), 'model': config['model'], 'method': config['method'],
                     'drop_rate': config['drop_rate'], 'attention_blocks': json.dumps(config['attention_blocks']),
                     'attention_drop': json.dumps(config['attention_drop']), 'top1_pct': summary['top1_pct'],
                     'top5_pct': summary['top5_pct'], 'images': summary['images'],
                     'mean_estimated_gmacs': summary['mean_estimated_macs']/1e9,
                     'metric_kind': summary['metric_kind'],
                     'metadata_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                     'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                     'preprocessing': config['preprocessing'], 'quantization': config['quantization'],
                     'weights_sha256': meta['weights_sha256'], 'class_map_sha256': meta['class_map_sha256']})
    args.output.mkdir(parents=True)
    with (args.output/'selected_runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for row in rows:
        label = f'{row["model"]}: {row["method"]}, EDPG {row["drop_rate"]:g}, ATS {row["attention_drop"]}'
        ax.scatter(row['mean_estimated_gmacs'], row['top1_pct'], label=label)
    ax.set(xlabel='Analytical dense-equivalent compute (GMACs)', ylabel='Top-1 accuracy (%)', title='Selected completed software evaluations')
    if any(r['metric_kind'] == 'random_initialization_smoke_test' for r in rows):
        ax.set_title('Random-initialization smoke tests — not research accuracy')
    ax.grid(alpha=.2); ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    save_figure(fig, args.output/'accuracy_compute')
    print(args.output)


if __name__ == '__main__':
    main()
