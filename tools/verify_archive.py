#!/usr/bin/env python3
"""Verify complete notebook reconstruction, exports, and optional byte-exact local copies."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from restore_notebook import restore
from archive_experiments import digest_file


def verify(repo, local=False, original_source=None):
    manifest = json.loads((repo / 'archive/manifest.json').read_text())
    checked = 0
    for entry in manifest['files']:
        if local:
            path = repo / entry['local_copy']
            if digest_file(path) != entry['sha256']:
                raise AssertionError(f'Local archive hash differs: {entry["source"]}')
        if original_source:
            if digest_file(original_source / entry['source']) != entry['sha256']:
                raise AssertionError(f'Original source changed: {entry["source"]}')
        if entry['kind'] == 'notebook':
            nb = restore(entry['source'], repo)
            canonical = json.dumps(nb, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
            if 'notebook_semantic_sha256' in entry:
                assert hashlib.sha256(canonical).hexdigest() == entry['notebook_semantic_sha256'], entry['source']
            if local:
                assert nb == json.loads((repo / entry['local_copy']).read_text()), entry['source']
            source = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'] if c['cell_type'] == 'code')
            tree = ast.parse((repo / entry['python_export']).read_text())
            assert isinstance(tree.body[0], ast.If), entry['source']
            exported = ast.Module(body=tree.body[1:], type_ignores=[])
            assert ast.dump(exported) == ast.dump(ast.parse(source)), entry['source']
            checked += 1
        elif 'portable_copy' in entry:
            assert digest_file(repo / entry['portable_copy']) == entry['sha256'], entry['source']
    result = {'files_accounted_for': len(manifest['files']), 'notebooks_reconstructed_and_code_verified': checked,
              'local_byte_hashes_verified': local, 'original_sources_verified': bool(original_source)}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--local', action='store_true')
    p.add_argument('--original-source', type=Path)
    args = p.parse_args()
    verify(args.repo, args.local, args.original_source)
