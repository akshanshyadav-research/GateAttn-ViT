#!/usr/bin/env python3
"""Verify complete notebook reconstruction, exports, and optional byte-exact local copies."""
import argparse
import ast
import hashlib
import json
import io
import subprocess
from pathlib import Path
from restore_notebook import restore
from archive_experiments import digest_file


def verify_tracked_originals(repo, entries):
    """Check every original file is indexed as its exact SHA-256 Git LFS object."""
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=repo).decode().split('\0'))
    missing = [e['local_copy'] for e in entries if e['local_copy'] not in tracked]
    if missing:
        raise AssertionError(f'Original files missing from Git: {missing[:10]}')
    queries = ''.join(':' + e['local_copy'] + '\n' for e in entries).encode()
    raw = subprocess.run(['git', 'cat-file', '--batch'], cwd=repo, input=queries, stdout=subprocess.PIPE, check=True).stdout
    stream = io.BytesIO(raw)
    for entry in entries:
        header = stream.readline().decode().strip().split()
        if len(header) != 3 or header[1] != 'blob':
            raise AssertionError(f'Cannot read indexed file: {entry["local_copy"]}')
        pointer = stream.read(int(header[2])).decode()
        assert stream.read(1) == b'\n'
        expected = f'version https://git-lfs.github.com/spec/v1\noid sha256:{entry["sha256"]}\nsize {entry["bytes"]}\n'
        if pointer != expected:
            raise AssertionError(f'Tracked LFS identity differs from original: {entry["source"]}')
    return len(entries)


def verify(repo, local=False, original_source=None, git_tracked=False):
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
    tracked_count = verify_tracked_originals(repo, manifest['files']) if git_tracked else None
    result = {'original_files_verified_in_git_lfs': tracked_count, 'files_accounted_for': len(manifest['files']), 'notebooks_reconstructed_and_code_verified': checked,
              'local_byte_hashes_verified': local, 'original_sources_verified': bool(original_source)}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--local', action='store_true', help='Verify byte-exact originals (run git lfs pull first)')
    p.add_argument('--git-tracked', action='store_true', help='Verify every original file is tracked with the correct LFS identity')
    p.add_argument('--original-source', type=Path)
    args = p.parse_args()
    verify(args.repo, args.local, args.original_source, args.git_tracked)
