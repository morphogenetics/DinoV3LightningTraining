#!/usr/bin/env python3
"""
Streaming tiled dataset for DINOv3 SSL training.
Reproduces the tiling geometry from MemoryEfficientImageFolder (the notebook's
classification dataset) but:
  - drops class labels (SSL doesn't need them; folders are just walked for images)
  - implements get_image_data()/get_target(), the interface contract shared by
    CSVDataset / HuggingFaceDataset / CustomImageDataset ("Method expected by
    DINOv3 data loading pipeline")
  - never writes tiles to disk -- everything is computed at access time

Compared to export_ssl_tiles.py (disk-materialized), this trades a bit of
CPU (re-decoding + re-resizing the source image on every access, since PIL
doesn't cache) for zero extra storage and zero duplicate-tile bookkeeping.
If your source images are large/slow to decode and you have workers to
spare, __getitem__ overhead is usually hidden by DataLoader parallelism --
but if it becomes a bottleneck, the disk-export version is the fallback.
"""

import os
import io
import logging
from typing import Callable, Optional, Tuple, List, Union

from PIL import Image
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dinov3")


class StreamingTiledDataset(Dataset):
    """
    Streaming (non-materialized) tiled dataset for SSL pretraining.

    Usage examples (once registered in the dataset dispatcher -- see note
    at the bottom of this file):
    - StreamingTiled:root=/path/to/images/
    - StreamingTiled:root=/path/to/images/:tile_size=272:tile_step=90
    """

    def __init__(
        self,
        root: Union[str, List[str]] = "../Datasets/composite/SLIDE-0018/",
        tile_size: int = 272,
        tile_step: int = 90,
        resize_factor: int = 3,
        max_width: int = 8000,
        threshold: int = 45,
        min_filesize_bytes: int = 2000,
        supported_extensions: Tuple[str, ...] = ('.tiff', '.tif', '.png', '.jpg', '.jpeg'),
        recursive: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        encode_format: str = "JPEG",
        encode_quality: int = 95,
    ):
        self.root_dirs = [root] if isinstance(root, str) else list(root)
        self.tile_size = tile_size
        self.tile_step = tile_step
        self.resize_factor = resize_factor
        self.max_width = max_width
        self.threshold = threshold
        self.min_filesize_bytes = min_filesize_bytes
        self.supported_extensions = tuple(e.lower() for e in supported_extensions)
        self.recursive = recursive
        self.transform = transform
        self.target_transform = target_transform
        self.encode_format = encode_format
        self.encode_quality = encode_quality

        # --- discover source images + sizes (metadata only, one open() per file) ---
        self.image_metadata = {}  # img_path -> (orig_w, orig_h)
        for current_root in self.root_dirs:
            if not os.path.isdir(current_root):
                logger.warning(f"Skipping {current_root}: not a valid directory.")
                continue
            walker = os.walk(current_root) if self.recursive else [(current_root, [], os.listdir(current_root))]
            for dirpath, _, filenames in walker:
                for fname in sorted(filenames):
                    if not fname.lower().endswith(self.supported_extensions):
                        continue
                    img_path = os.path.join(dirpath, fname)
                    if os.path.getsize(img_path) <= self.min_filesize_bytes:
                        continue
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                    except Exception:
                        continue
                    if width < self.threshold:
                        continue
                    if self.max_width is not None and width > self.max_width:
                        continue
                    self.image_metadata[img_path] = (width, height)

        # --- build flat tile index: (img_path, tile_idx, left, top) ---
        self.tile_metadata = []
        for img_path, (orig_w, orig_h) in self.image_metadata.items():
            new_w = orig_w * self.resize_factor
            n_tiles = (new_w - self.tile_size) // self.tile_step + 1
            for tile_idx in range(max(n_tiles, 0)):
                left = tile_idx * self.tile_step
                if left + self.tile_size > new_w:
                    left = new_w - self.tile_size
                self.tile_metadata.append((img_path, tile_idx, left, 0))

        logger.info(
            f"StreamingTiledDataset: {len(self.image_metadata)} source images, "
            f"{len(self.tile_metadata)} tiles (streamed, not materialized)"
        )
        if len(self.tile_metadata) == 0:
            logger.warning(f"No tiles found under {self.root_dirs} -- check paths/extensions/size filters.")

    def __len__(self) -> int:
        return len(self.tile_metadata)

    def _extract_tile(self, idx: int) -> Image.Image:
        """Core lazy-loading logic: open source image, resize, crop the tile.
        This is the one piece of work every access path (transform-based
        __getitem__ AND raw-bytes get_image_data) funnels through."""
        img_path, tile_idx, left, top = self.tile_metadata[idx]
        with Image.open(img_path) as image:
            image = image.convert("RGB")
            new_w = image.width * self.resize_factor
            new_h = image.height * self.resize_factor
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)
            tile = image.crop((left, top, left + self.tile_size, top + self.tile_size))
        return tile

    def __getitem__(self, idx: int) -> Tuple:
        """Standard PyTorch-style access: returns (transformed_tile, dummy_target).
        NOTE: if the SSL data module drives training via get_image_data()
        instead of __getitem__ (as the other three loaders' docstrings
        suggest -- see module note below), this path may not even be used
        during actual training; keep it for standalone/debug use regardless."""
        tile = self._extract_tile(idx)
        if self.transform:
            tile = self.transform(tile)
        target = idx
        if self.target_transform:
            target = self.target_transform(target)
        return tile, target

    def get_image_data(self, index: int) -> bytes:
        """Method expected by DINOv3 data loading pipeline.
        Unlike the file-backed loaders, there's no raw file to read for a
        single tile -- crop it, then re-encode to bytes in memory so
        downstream code can decode + apply its own multi-crop augmentation
        exactly as it would for a real file."""
        try:
            tile = self._extract_tile(index)
            buf = io.BytesIO()
            tile.save(buf, format=self.encode_format, quality=self.encode_quality)
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Error extracting tile at index {index}: {e}")
            return b''

    def get_target(self, index: int) -> int:
        """Method expected by DINOv3 data loading pipeline. Dummy target for SSL."""
        return index

    def get_image_paths(self) -> List[str]:
        """Source image path for each tile (not unique per tile -- multiple
        tiles share a source image). Useful for provenance/debugging."""
        return [t[0] for t in self.tile_metadata]


# Backward-compatible alias matching the naming convention of the other loaders
StreamingTiledSSLDataset = StreamingTiledDataset


if __name__ == "__main__":
    import tempfile
    import numpy as np
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        src_dir = Path(tmp) / "images"
        src_dir.mkdir()
        rng = np.random.RandomState(0)
        for i in range(3):
            arr = rng.randint(0, 255, (100, 400, 3), dtype=np.uint8)
            Image.fromarray(arr).save(src_dir / f"img_{i:03d}.png")

        ds = StreamingTiledDataset(
            root=str(src_dir),
            tile_size=64, tile_step=32,
            threshold=10, min_filesize_bytes=10,
        )
        print(f"Dataset length: {len(ds)}")
        assert len(ds) > 0

        # __getitem__ path
        tile, target = ds[0]
        assert tile.size == (64, 64), f"expected 64x64 tile, got {tile.size}"
        assert target == 0
        print(f"[OK] __getitem__ returns {tile.size} PIL tile, target={target}")

        # get_image_data path -- round-trip through bytes
        raw_bytes = ds.get_image_data(5)
        assert len(raw_bytes) > 0
        decoded = Image.open(io.BytesIO(raw_bytes))
        decoded.load()
        assert decoded.size == (64, 64)
        print(f"[OK] get_image_data returns {len(raw_bytes)} bytes, "
              f"decodes back to {decoded.size} tile")

        assert ds.get_target(5) == 5
        print("[OK] get_target returns dummy index")

        paths = ds.get_image_paths()
        assert len(paths) == len(ds)
        print(f"[OK] get_image_paths returns {len(paths)} entries")

        print("\n[OK] all smoke tests passed")

    # --- registration note ---
    # This class still needs to be wired into whatever function parses the
    # "Prefix:key=value" dataset_path string (confirmed to live in
    # dinov3/dinov3/data/loaders.py's make_dataset(), based on train.py's
    # "CustomTIFF:root=..." example -- not yet seen). Once you find that
    # file, register a prefix (e.g. "StreamingTiled") pointing at this
    # class, following whatever pattern CustomTIFF/CSV/HuggingFace use there.
