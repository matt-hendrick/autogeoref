"""Synchronization for Pillow's process-global image safety settings."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from PIL import Image

_IMAGE_MAX_PIXELS_LOCK = threading.RLock()


@contextmanager
def unlimited_image_pixels() -> Iterator[None]:
    """Temporarily lift Pillow's global pixel cap while a decode is in progress."""
    with _IMAGE_MAX_PIXELS_LOCK:
        saved_cap = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            yield
        finally:
            Image.MAX_IMAGE_PIXELS = saved_cap
