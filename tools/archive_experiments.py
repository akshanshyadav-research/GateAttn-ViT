#!/usr/bin/env python3
"""Preserve every source file; export all notebooks and losslessly store outputs.

Run once on a NEW repository archive. Never executes notebook code.
"""
from __future__ import annotations
import argparse
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def notebook_metadata(nb):
    source = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'] if c['cell_type'] == 'code')
    tree = ast.parse(source)
    config = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'patch_drop_blocks', 'attention_pruning', 'drop_patch_percentage_entropy', 'keep_percentage', 'drop_rate'}:
                    try:
                        config[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    models = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'create_model' and node.args:
            if isinstance(node.args[0], ast.Constant):
                models.append(node.args[0].value)
    return source, config, list(dict.fromkeys(models)), sorted({n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))})


def archive(source, repo):
    source, repo = source.resolve(), repo.resolve()
    target = repo / 'archive'
    if (target / 'manifest.json').exists():
        raise SystemExit('Archive already exists; use verify_archive.py rather than overwriting provenance.')
    files = sorted(p for p in source.rglob('*') if p.is_file() and not p.is_relative_to(repo) and '.git' not in p.parts)
    directories = sorted(str(p.relative_to(source)) for p in source.rglob('*') if p.is_dir() and not p.is_relative_to(repo) and '.git' not in p.parts)
    entries, notebooks, metric_rows = [], [], []
    pattern = re.compile(r'Accuracy after processing\s+(\d+)\s+images:\s*([\d.]+)%')
    for number, path in enumerate(files):
        rel = path.relative_to(source)
        entry = {'source': str(rel), 'bytes': path.stat().st_size, 'sha256': digest_file(path)}
        if path.suffix == '.ipynb':
            original = target / 'originals' / rel
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, original)
            nb = json.loads(path.read_text())
            entry['notebook_semantic_sha256'] = hashlib.sha256(json.dumps(nb, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
            source_code, config, models, functions = notebook_metadata(nb)
            errors, output_refs = [], {}
            for i, cell in enumerate(nb['cells']):
                if cell.get('outputs'):
                    outputs = cell['outputs']
                    for j, output in enumerate(outputs):
                        if output.get('output_type') == 'error':
                            errors.append({'cell': i, 'name': output.get('ename'), 'message': output.get('evalue')})
                        text = ''.join(output.get('text', []))
                        for match in pattern.finditer(text):
                            metric_rows.append({'source': str(rel), 'cell': i, 'output': j, 'processed_images': int(match[1]), 'accuracy_pct': float(match[2]), 'model_from_source': '|'.join(models), 'entropy_drop_pct_from_source': config.get('drop_patch_percentage_entropy', ''), 'attention_drop_from_source': config.get('attention_pruning', ''), 'blocks_from_source': json.dumps(config.get('patch_drop_blocks', [])), 'checkpoint': '.ipynb_checkpoints' in rel.parts})
                    raw = json.dumps(outputs, ensure_ascii=False, separators=(',', ':')).encode()
                    refs = []
                    for start in range(0, len(raw), 4 * 1024 * 1024):
                        chunk = raw[start:start + 4 * 1024 * 1024]
                        key = hashlib.sha256(chunk).hexdigest()
                        blob = target / 'output_objects' / key[:2] / (key + '.json.gz')
                        if not blob.exists():
                            blob.parent.mkdir(parents=True, exist_ok=True)
                            blob.write_bytes(gzip.compress(chunk, mtime=0))
                        refs.append(key)
                    output_refs[str(i)] = refs
                    cell['outputs'] = []
            stripped = target / 'notebooks' / rel
            write_json(stripped, nb)
            refs_path = target / 'output_index' / rel.with_suffix('.json')
            write_json(refs_path, output_refs)
            script = target / 'python' / rel.with_suffix('.py')
            script.parent.mkdir(parents=True, exist_ok=True)
            parts = ['#!/usr/bin/env python3\n', '# Original notebook: ' + str(rel) + '\n', '# Archival export. See docs/ARCHIVE.md before executing.\n', 'if __name__ != "__main__":\n', '    raise ImportError("Archival experiment: use the gateattn package for imports.")\n\n']
            for i, cell in enumerate(nb['cells']):
                text = ''.join(cell.get('source', []))
                parts.append('# %% Cell ' + str(i) + ' (' + cell['cell_type'] + ')\n')
                if cell['cell_type'] == 'code':
                    parts.append(text + '\n\n')
                else:
                    parts.extend('# ' + line + '\n' for line in text.splitlines())
            script.write_text(''.join(parts))
            compile(script.read_text(), str(script), 'exec')
            entry.update(kind='notebook', local_copy=str(original.relative_to(repo)), portable_copy=str(stripped.relative_to(repo)), python_export=str(script.relative_to(repo)), output_index=str(refs_path.relative_to(repo)))
            notebooks.append({'source': str(rel), 'config': config, 'models_from_source': models, 'functions': functions, 'cells': len(nb['cells']), 'output_cells': len(output_refs), 'saved_errors': errors, 'checkpoint': '.ipynb_checkpoints' in rel.parts, 'code_sha256': hashlib.sha256(source_code.encode()).hexdigest()})
        else:
            local = repo / 'local_assets' / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, local)
            entry.update(kind='asset', local_copy=str(local.relative_to(repo)))
            if path.suffix in {'.txt', '.py', '.md', '.csv', '.yaml', '.yml', '.toml', '.sh', '.v', '.sv', '.cpp', '.h', '.tcl'}:
                portable = target / 'files' / rel
                portable.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, portable)
                entry['portable_copy'] = str(portable.relative_to(repo))
        entries.append(entry)
        if number % 500 == 0:
            print(f'Archived {number + 1}/{len(files)} files', flush=True)
    write_json(target / 'manifest.json', {'format_version': 1, 'source_root_label': source.name, 'directories': directories, 'files': entries})
    write_json(target / 'notebook_catalog.json', notebooks)
    out = repo / 'results' / 'extracted'
    out.mkdir(parents=True, exist_ok=True)
    for row in metric_rows:
        catalog = next(n for n in notebooks if n['source'] == row['source'])
        row['notebook_has_saved_error'] = bool(catalog['saved_errors'])
    if metric_rows:
        with (out / 'notebook_accuracy_history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
            writer.writeheader(); writer.writerows(metric_rows)
    summary = {'source_files': len(entries), 'notebooks': len(notebooks), 'checkpoint_notebooks': sum(n['checkpoint'] for n in notebooks), 'assets': len(entries) - len(notebooks), 'source_bytes': sum(e['bytes'] for e in entries), 'accuracy_observations': len(metric_rows), 'notebooks_with_saved_errors': sum(bool(n['saved_errors']) for n in notebooks)}
    write_json(target / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    archive(args.source, args.repo)
