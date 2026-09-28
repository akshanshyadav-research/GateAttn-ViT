"""Image-patch scoring and selection, derived from the archived notebooks."""
from __future__ import annotations
import math
import cv2
import numpy as np


def patch_entropy(image: np.ndarray, patch_size: int = 16, mode: str = 'gray') -> np.ndarray:
    """Shannon entropy in bits; RGB mode pools all channel values, as in notebooks.

    Input is uint8 RGB or grayscale, already spatially transformed for the model.
    """
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError('Expected a uint8 RGB or grayscale image.')
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError('RGB input must have three channels.')
    if mode not in {'gray', 'rgb'}:
        raise ValueError('Entropy mode must be gray or rgb.')
    if mode == 'gray' and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = image.shape[:2]
    if patch_size <= 0 or h % patch_size or w % patch_size:
        raise ValueError('Image dimensions must be divisible by patch_size.')
    values = []
    for row in range(0, h, patch_size):
        for col in range(0, w, patch_size):
            patch = image[row:row + patch_size, col:col + patch_size]
            counts = np.bincount(patch.reshape(-1), minlength=256)
            probabilities = counts[counts > 0] / patch.size
            values.append(-np.sum(probabilities * np.log2(probabilities)))
    return np.asarray(values, dtype=np.float64)


def keep_count(count: int, drop_rate: float) -> int:
    if count < 1 or not math.isfinite(drop_rate) or not 0 <= drop_rate < 1:
        raise ValueError('Require at least one patch and 0 <= drop_rate < 1.')
    return max(1, int((1 - drop_rate) * count))


def select_mask(scores: np.ndarray, drop_rate: float, policy: str = 'legacy') -> np.ndarray:
    """Return patch-only mask. Legacy reproduces np.argsort top-k selection.

    stable: exact count with lower spatial indices winning ties.
    threshold: paper-style threshold, retaining all ties at the cutoff.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('Scores must be a finite one-dimensional array.')
    k = keep_count(len(scores), drop_rate)
    mask = np.zeros(len(scores), dtype=bool)
    if policy == 'legacy':
        mask[np.argsort(scores)[-k:]] = True
    elif policy == 'stable':
        mask[np.argsort(-scores, kind='stable')[:k]] = True
    elif policy == 'threshold':
        mask = scores >= np.sort(scores)[-k]
    else:
        raise ValueError(f'Unknown entropy selection policy: {policy}')
    return mask


def baseline_mask(grid_size: int, drop_rate: float, method: str, rng=None) -> np.ndarray:
    """Random or center selection. center-square matches the original notebooks.

    center ranks by distance from the center to keep an exact patch count.
    center-square keeps floor(sqrt(k))**2 patches, which may exceed requested drop.
    """
    k = keep_count(grid_size ** 2, drop_rate)
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    if method == 'random':
        if rng is None:
            raise ValueError('Pass an explicitly seeded NumPy generator.')
        mask.flat[rng.choice(grid_size ** 2, k, replace=False)] = True
    elif method == 'center-square':
        side = int(np.sqrt(k))
        start = (grid_size - side) // 2
        mask[start:start + side, start:start + side] = True
    elif method == 'center':
        yy, xx = np.indices(mask.shape)
        distance = (xx - (grid_size - 1) / 2) ** 2 + (yy - (grid_size - 1) / 2) ** 2
        mask.flat[np.argsort(distance.ravel(), kind='stable')[:k]] = True
    else:
        raise ValueError(f'Unknown baseline method: {method}')
    return mask.ravel()
