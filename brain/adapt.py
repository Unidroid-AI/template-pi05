# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map observations to a pi0.5 DROID policy and back to absolute joints."""

from __future__ import annotations

import numpy as np

IMAGE_HW = (224, 224)

BATCH_SIZES = (1, 4, 16)
"""Row counts the model is compiled for. A call is padded up to the next one."""


def padded_rows(rows: int) -> int:
    """Return the compiled batch size that holds ``rows`` rows.

    The size depends only on ``rows``, so a row is always computed in the same
    program no matter how many calls the server has merged or in what order.

    Args:
        rows: Number of real rows, at most ``BATCH_SIZES[-1]``.
    """
    for size in BATCH_SIZES:
        if rows <= size:
            return size
    raise ValueError(f"{rows} rows exceed the largest batch size {BATCH_SIZES[-1]}")


def row_slices(rows: int) -> list[slice]:
    """Split ``rows`` into slices of at most ``BATCH_SIZES[-1]`` rows."""
    limit = BATCH_SIZES[-1]
    return [slice(start, min(start + limit, rows)) for start in range(0, rows, limit)]


def stack_padded(items: list, size: int):
    """Stack nested dicts of arrays on a new axis 0 and pad to ``size`` rows.

    Pad rows repeat the first row, so they never feed zeros or NaNs to the model.

    Args:
        items: One nested dict of arrays per row, all with the same structure.
        size: Row count after padding, at least ``len(items)``.
    """
    if len(items) > size:
        raise ValueError(f"{len(items)} rows do not fit in a batch of {size}")
    padded = list(items) + [items[0]] * (size - len(items))
    return _stack(padded)


def _stack(items: list):
    first = items[0]
    if isinstance(first, dict):
        return {key: _stack([item[key] for item in items]) for key in first}
    return np.stack([np.asarray(item) for item in items], axis=0)


def resize_with_pad(image: np.ndarray, height: int = IMAGE_HW[0], width: int = IMAGE_HW[1]) -> np.ndarray:
    """Resize one HWC image to ``height`` x ``width`` without stretching.

    Scale uniformly and pad the short side with zeros, matching
    ``openpi_client.image_tools.resize_with_pad``. An image that is already
    the target size is returned unchanged.

    Args:
        image: uint8 image, shape [H, W, C].
        height: Target height [pixels].
        width: Target width [pixels].
    """
    array = np.asarray(image)
    if array.shape[-3:-1] == (height, width):
        return array
    from PIL import Image

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    current_height, current_width = array.shape[0], array.shape[1]
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = Image.fromarray(array).resize((resized_width, resized_height), resample=Image.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    canvas.paste(resized, (pad_width, pad_height))
    return np.asarray(canvas)


def joint_chunk(actions: np.ndarray, horizon: int) -> np.ndarray:
    """Keep absolute joint targets [rad] and snap the gripper.

    The pi0.5 DROID joint-position config adds the observed joints back onto
    the first seven columns, so those values are already absolute. The eighth
    column is the gripper. A value above 0.5 closes it.

    Args:
        actions: Policy actions, shape [H, 8] or wider.
        horizon: Number of steps to keep.

    Returns:
        Actions, shape [horizon, 8]. The last column is 0 open or 1 closed.
    """
    raw = np.asarray(actions, dtype=np.float32)
    chunk = raw[:horizon, :8].copy()
    if chunk.shape != (horizon, 8):
        raise ValueError(f"expected at least {(horizon, 8)} actions, got {raw.shape}")
    chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
    return chunk
