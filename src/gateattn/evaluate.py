"""Reproducible single-image evaluation with explicit input and checkpoint metadata."""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from .complexity import estimate_macs
from .data import load_class_map, preprocess, samples
from .entropy import baseline_mask, patch_entropy, select_mask
from .model import GateAttnViT
from .quantization import fixed_point, quantize_weights


@dataclass
class ExperimentConfig:
    model: str = 'vit_base_patch16_224'
    method: str = 'entropy'
    drop_rate: float = 0.15
    entropy_mode: str = 'gray'
    entropy_policy: str = 'legacy'
    attention_blocks: tuple = (3, 6, 9)
    attention_drop: float | list = 0.15
    attention_policy: str = 'quantile'
    preprocessing: str = 'legacy'
    quantization: str = 'none'
    word_bits: int = 16
    fractional_bits: int = 8
    layer_norm_eps: float | None = 1e-5
    seed: int = 0
    random_init: bool = False
    checkpoint: str | None = None

    def validate(self):
        if self.method not in {'entropy', 'random', 'center', 'center-square', 'none'}:
            raise ValueError('Unsupported patch selection method.')
        if not np.isfinite(self.drop_rate) or not 0 <= self.drop_rate < 1:
            raise ValueError('drop_rate must be in [0, 1).')
        if self.method == 'none' and self.drop_rate != 0:
            raise ValueError('method=none requires drop_rate=0.')
        if self.entropy_mode not in {'gray', 'rgb'} or self.entropy_policy not in {'legacy', 'stable', 'threshold'}:
            raise ValueError('Invalid entropy mode or policy.')
        if self.preprocessing not in {'legacy', 'timm'} or self.quantization not in {'none', 'notebook-fx16'}:
            raise ValueError('Invalid preprocessing or quantization policy.')
        if self.layer_norm_eps is not None and (not np.isfinite(self.layer_norm_eps) or self.layer_norm_eps <= 0):
            raise ValueError('layer_norm_eps must be positive or null for the checkpoint default.')
        if self.random_init and self.checkpoint:
            raise ValueError('Choose either a checkpoint or random initialization.')
        fixed_point(torch.tensor(0.), self.word_bits, self.fractional_bits)
        return self


def config_from_args(args):
    config = json.loads(Path(args.config).read_text()) if getattr(args, 'config', None) else {}
    allowed = {f.name for f in fields(ExperimentConfig)}
    unknown = config.keys() - allowed
    if unknown:
        raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
    for key in allowed:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    return ExperimentConfig(**config).validate()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(config, device, capture_attention=False):
    import timm
    config.validate()
    seed_everything(config.seed)
    kwargs = {'pretrained': not config.random_init and not config.checkpoint}
    if config.checkpoint:
        kwargs['checkpoint_path'] = config.checkpoint
    backbone = timm.create_model(config.model, **kwargs)
    if config.layer_norm_eps is not None:
        for module in backbone.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.eps = config.layer_norm_eps
    if config.quantization == 'notebook-fx16':
        quantize_weights(backbone, config.word_bits, config.fractional_bits)
    model = GateAttnViT(backbone, config.attention_blocks, config.attention_drop,
                       config.attention_policy, capture_attention=capture_attention)
    return model.to(device).eval()


def prepare(path, model, config, rng):
    size = model.backbone.patch_embed.img_size
    patch = model.backbone.patch_embed.patch_size
    if size[0] != size[1] or patch[0] != patch[1]:
        raise ValueError('Only square images and patches are supported.')
    tensor, entropy_image, visual = preprocess(path, model.backbone, config.preprocessing, size[0])
    if config.entropy_mode == 'rgb' and config.preprocessing == 'legacy':
        # Original RGB entropy variants use bilinear resized RGB, separate from inference.
        import cv2
        from PIL import Image
        with Image.open(path) as image:
            entropy_image = cv2.resize(np.asarray(image.convert('RGB')), size[::-1], interpolation=cv2.INTER_LINEAR)
    scores = patch_entropy(entropy_image, patch[0], config.entropy_mode)
    if config.method == 'entropy':
        mask = select_mask(scores, config.drop_rate, config.entropy_policy)
    elif config.method == 'none':
        mask = np.ones(len(scores), dtype=bool)
    else:
        mask = baseline_mask(size[0] // patch[0], config.drop_rate, config.method, rng)
    if config.quantization == 'notebook-fx16':
        tensor = fixed_point(tensor, config.word_bits, config.fractional_bits)
    return tensor, mask, scores, visual


def environment():
    versions = {}
    for name in ['torch', 'torchvision', 'timm', 'numpy', 'opencv-python', 'Pillow', 'matplotlib']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not found'
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions}


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def weight_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_evaluation(config, data, class_map, output, device='cpu', max_images=None):
    config.validate()
    if max_images is not None and max_images < 1:
        raise ValueError('max_images must be positive.')
    mapping = load_class_map(class_map)
    dataset = list(samples(data, mapping))
    if max_images is not None:
        dataset = dataset[:max_images]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    model = build_model(config, device)
    if max(mapping.values()) >= model.backbone.num_classes:
        raise ValueError('Class map contains an index outside the model output range.')
    output.mkdir(parents=True)
    metadata = {'status': 'running', 'config': asdict(config), 'environment': environment(),
                'dataset_root': str(Path(data).resolve()), 'class_map_sha256': hash_file(class_map),
                'device': str(device), 'requested_images': len(dataset),
                'weights_sha256': weight_digest(model),
                'pretrained_config': getattr(model.backbone, 'pretrained_cfg', {}),
                'metric_kind': 'random_initialization_smoke_test' if config.random_init else 'new_evaluation'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    rng = np.random.default_rng(config.seed)
    correct = correct5 = 0
    mac_total = forward_seconds = 0.
    started = time.perf_counter()
    try:
        with (output / 'predictions.jsonl').open('w') as stream, torch.inference_mode():
            for index, (path, target) in enumerate(dataset, 1):
                tensor, mask, _, _ = prepare(path, model, config, rng)
                tensor = tensor.to(device)
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                tick = time.perf_counter()
                logits = model(tensor, mask)
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                duration = time.perf_counter() - tick
                forward_seconds += duration
                top = logits[0].topk(min(5, logits.shape[-1])).indices.cpu().tolist()
                correct += int(top[0] == target)
                correct5 += int(target in top)
                b = model.backbone
                macs = estimate_macs(model.token_counts, b.embed_dim, b.blocks[0].mlp.fc1.out_features,
                                     b.patch_embed.img_size[0], b.patch_embed.patch_size[0], b.num_classes)
                mac_total += macs
                record = {'image': str(path.relative_to(data)), 'image_sha256': hash_file(path), 'target': target,
                          'top5': top, 'correct': top[0] == target, 'input_patches_kept': int(mask.sum()),
                          'block_input_tokens_including_cls': model.token_counts,
                          'final_patches_kept': len(model.retained_indices), 'estimated_macs': macs,
                          'forward_seconds': duration}
                stream.write(json.dumps(record) + '\n')
                if index % 100 == 0 or index == len(dataset):
                    stream.flush()
                    print(f'{index}/{len(dataset)} images, top-1 {100 * correct / index:.2f}%', flush=True)
        result = {'status': 'complete', 'images': len(dataset), 'top1_pct': 100 * correct / len(dataset),
                  'top5_pct': 100 * correct5 / len(dataset), 'mean_estimated_macs': mac_total / len(dataset),
                  'mean_forward_seconds': forward_seconds / len(dataset), 'wall_seconds': time.perf_counter() - started,
                  'timing_scope': 'single-image model forward only; no warmup; not FPGA latency',
                  'metric_kind': metadata['metric_kind']}
        (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    return result


def add_config_arguments(parser):
    parser.add_argument('--config', help='JSON ExperimentConfig file')
    parser.add_argument('--model')
    parser.add_argument('--method', choices=['entropy', 'random', 'center', 'center-square', 'none'])
    parser.add_argument('--drop-rate', type=float)
    parser.add_argument('--attention-blocks', type=int, nargs='*')
    parser.add_argument('--attention-drop', type=float)
    parser.add_argument('--preprocessing', choices=['legacy', 'timm'])
    parser.add_argument('--quantization', choices=['none', 'notebook-fx16'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--random-init', action='store_true', default=None, help='Smoke test only; do not report accuracy as a research result')
    parser.add_argument('--device', default='cpu')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--class-map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-images', type=int)
    args = parser.parse_args()
    result = run_evaluation(config_from_args(args), args.data, args.class_map, args.output, args.device, args.max_images)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
