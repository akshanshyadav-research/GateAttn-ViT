#!/usr/bin/env python3
"""Summarize saved observations without asserting completion or deduplicating runs."""
import argparse
import collections
import csv
import json
from pathlib import Path
import re


def write_csv(path, rows):
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def extract(repo):
    folder = repo / 'results/extracted'
    with (folder / 'notebook_accuracy_history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    groups = collections.defaultdict(list)
    for row in history:
        groups[row['source']].append(row)
    summaries = []
    for source, rows in groups.items():
        final = dict(rows[-1])
        final.update(observations=len(rows), maximum_images_observed=max(int(r['processed_images']) for r in rows),
                     status='saved_observation_only_completion_unverified')
        summaries.append(final)
    write_csv(folder / 'notebook_last_observations.csv', summaries)
    records = []
    pattern = re.compile(r'Accuracy:\s*([\d.]+),\s*processed_image:\s*(\d+)')
    for path in sorted((repo / 'archive/files').rglob('*.txt')):
        run_segment, previous = 0, -1
        for line_number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
            match = pattern.search(line)
            if not match:
                continue
            count = int(match[2])
            if count <= previous:
                run_segment += 1
            records.append({'source': str(path.relative_to(repo / 'archive/files')), 'line': line_number,
                            'segment_inferred_from_count_reset': run_segment, 'processed_images': count,
                            'accuracy_pct': float(match[1]), 'model_config': 'not inferred from filename'})
            previous = count
    write_csv(folder / 'text_log_accuracy_history.csv', records)
    print(f'{len(history)} notebook observations, {len(summaries)} notebook summaries, {len(records)} text-log observations.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    extract(p.parse_args().repo)
