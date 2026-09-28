"""Explicit class mappings and recorded spatial preprocessing."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import torch

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif'}


def load_class_map(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError('Class map must be a nonempty JSON object.')
    if all(isinstance(v, list) and v and isinstance(v[0], str) for v in data.values()):
        result = {value[0]: int(key) for key, value in data.items()}
    elif all(type(value) is int for value in data.values()):
        result = data
    else:
        raise ValueError('Use {class_directory: model_index} or ImageNet {index: [wnid, label]}.')
    if any(v < 0 for v in result.values()) or len(set(result.values())) != len(result):
        raise ValueError('Class indices must be nonnegative and unique.')
    return result


def samples(root, class_map):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset directory does not exist: {root}')
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError('No images found in dataset.')
    for path in paths:
        relative = path.relative_to(root)
        if len(relative.parts) < 2 or relative.parts[0] not in class_map:
            raise ValueError(f'No model class mapping for image: {relative}')
        yield path, class_map[relative.parts[0]]


def preprocess(path, model, policy='legacy', image_size=224):
    """Return model tensor and RGB image used for entropy on exactly its patch grid.

    Legacy keeps the notebook's nearest-neighbor RGB tensor and bilinear grayscale
    entropy branch. timm follows the checkpoint's normalization and spatial transform.
    """
    with Image.open(path) as source:
        rgb = np.asarray(source.convert('RGB'))
    if policy == 'legacy':
        visual = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        tensor = torch.from_numpy(visual.copy()).permute(2, 0, 1).float() / 255
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        entropy_image = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    elif policy == 'timm':
        from timm.data import create_transform, resolve_model_data_config
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        tensor = transform(Image.fromarray(rgb))
        mean = torch.tensor(config['mean'])[:, None, None]
        std = torch.tensor(config['std'])[:, None, None]
        visual = ((tensor * std + mean).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
        entropy_image = visual
    else:
        raise ValueError('Preprocessing must be legacy or timm.')
    return tensor.unsqueeze(0), entropy_image, visual
