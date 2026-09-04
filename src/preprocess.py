"""Small, dependency-light image preprocessing helpers for exported models."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize an image with unchanged aspect ratio and pad to ``new_shape``.

    Returns the padded RGB image, the resize ratio, and the (left, top) padding.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")

    target_h, target_w = new_shape
    height, width = image.shape[:2]
    ratio = min(target_w / width, target_h / height)
    resized_w = int(round(width * ratio))
    resized_h = int(round(height * ratio))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - resized_w
    pad_h = target_h - resized_h
    left = int(round(pad_w / 2 - 0.1))
    right = pad_w - left
    top = int(round(pad_h / 2 - 0.1))
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), ratio, (float(left), float(top))


def to_model_input(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to a float32 NHWC tensor in the [0, 1] range."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    return (image_rgb.astype(np.float32) / 255.0)[None, ...]
