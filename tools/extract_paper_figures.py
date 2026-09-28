#!/usr/bin/env python3
"""Extract the approved JSA graph outputs without executing notebook code."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=root / 'notebooks/jsa_paper_graphs.ipynb')
    parser.add_argument('--output', type=Path, default=root / 'figures/paper')
    args = parser.parse_args()
    manifest = json.loads((root / 'figures/paper/manifest.json').read_text())
    notebook = json.loads(args.notebook.read_text())
    cells = {}
    for cell in notebook['cells']:
        cell_id = cell.get('metadata', {}).get('id')
        if cell_id:
            if cell_id in cells:
                parser.error(f'Duplicate source cell ID: {cell_id}')
            cells[cell_id] = cell
    exports = []
    for item in manifest['figures']:
        try:
            cell = cells[item['source_cell_id']]
            output = cell['outputs'][item['source_output_index']]
            encoded = ''.join(output['data']['image/png'])
            data = base64.b64decode(''.join(encoded.split()), validate=True)
        except (KeyError, IndexError, ValueError) as exc:
            parser.error(f"Missing or invalid original output for {item['path']}: {exc}")
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            parser.error(f"Original output checksum mismatch: {item['path']}")
        exports.append((item['path'], data))
    # Validate every selected output before writing any files.
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in exports:
        (args.output / name).write_bytes(data)
    print(f'Extracted {len(exports)} verified original outputs to {args.output}')


if __name__ == '__main__':
    main()
