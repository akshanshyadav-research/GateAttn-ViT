#!/usr/bin/env python3
"""Restore all saved notebook outputs; optionally extract original figures."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def restore(relative, repo=REPO):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Use a notebook path relative to the original experiment folder.')
    nb = json.loads((repo / 'archive/notebooks' / relative).read_text())
    index = json.loads((repo / 'archive/output_index' / relative.with_suffix('.json')).read_text())
    for cell, refs in index.items():
        chunks = []
        for key in refs:
            chunk = gzip.decompress((repo / 'archive/output_objects' / key[:2] / (key + '.json.gz')).read_bytes())
            if hashlib.sha256(chunk).hexdigest() != key:
                raise ValueError(f'Corrupt output object: {key}')
            chunks.append(chunk)
        nb['cells'][int(cell)]['outputs'] = json.loads(b''.join(chunks))
    return nb


def extract_media(nb, destination):
    destination.mkdir(parents=True, exist_ok=True)
    extensions = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/svg+xml': '.svg', 'application/pdf': '.pdf'}
    count = 0
    for i, cell in enumerate(nb['cells']):
        for j, output in enumerate(cell.get('outputs', [])):
            for mime, ext in extensions.items():
                if mime in output.get('data', {}):
                    data = output['data'][mime]
                    if isinstance(data, list):
                        data = ''.join(data)
                    content = data.encode() if mime == 'image/svg+xml' else base64.b64decode(data)
                    (destination / f'cell_{i:03d}_output_{j:03d}{ext}').write_bytes(content)
                    count += 1
    return count


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', help='Original relative notebook path; quote paths containing spaces')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extract-media', type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; choose a new destination.')
    restored = restore(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(restored, indent=2, ensure_ascii=False) + '\n')
    if args.extract_media:
        print(f'Extracted {extract_media(restored, args.extract_media)} figures.')
    print(f'Restored {args.output}')
