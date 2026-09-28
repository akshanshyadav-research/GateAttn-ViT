#!/usr/bin/env python3
"""Create or run deterministic grids without modifying notebooks."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from gateattn.evaluate import ExperimentConfig, run_evaluation


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['entropy', 'combined', 'placement', 'baselines'], required=True)
    p.add_argument('--models', nargs='+', default=['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224'])
    p.add_argument('--rates', type=float, nargs='+', default=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--class-map', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--max-images', type=int)
    p.add_argument('--run', action='store_true', help='Otherwise only write configuration files')
    args = p.parse_args()
    if args.run and (not args.data or not args.class_map):
        p.error('--run requires --data and --class-map')
    if args.output.exists():
        p.error('Choose a new output directory.')
    args.output.mkdir(parents=True)
    configs = []
    for model in args.models:
        base = ExperimentConfig(model=model, attention_blocks=(), attention_drop=0, quantization='notebook-fx16')
        if args.suite == 'combined':
            for early, late in [(0, 0), (.1, .1), (.15, .1), (.1, .15), (.15, .15), (.15, .2), (.2, .15), (.2, .2), (.2, .25)]:
                configs.append(replace(base, drop_rate=early, attention_blocks=(3, 6, 9), attention_drop=late))
        else:
            for rate in args.rates:
                if args.suite == 'entropy':
                    configs.append(replace(base, drop_rate=rate))
                elif args.suite == 'placement':
                    configs.append(replace(base, drop_rate=rate))
                    for block in (0, 1, 2):
                        configs.append(replace(base, drop_rate=0, attention_blocks=(block,), attention_drop=rate))
                else:
                    for method in ('entropy', 'center-square', 'center', 'random'):
                        for seed in args.seeds if method == 'random' else [args.seeds[0]]:
                            configs.append(replace(base, drop_rate=rate, method=method, seed=seed))
    for i, config in enumerate(configs):
        config.validate()
        (args.output / f'{i:04d}.json').write_text(json.dumps(asdict(config), indent=2) + '\n')
        if args.run:
            run_evaluation(config, args.data, args.class_map, args.output / f'run_{i:04d}', args.device, args.max_images)
    print(f'{len(configs)} configurations written to {args.output}')
