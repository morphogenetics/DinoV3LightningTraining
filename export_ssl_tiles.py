"""
Materializes tiles from your classification notebook's tiling/preprocessing
logic out to actual image files on disk, for use as an unlabeled SSL
pretraining corpus.

Reuses the exact same tiling geometry as MemoryEfficientImageFolder (3x
resize, tile_size/tile_step sliding window) but:
  - drops class labels entirely (SSL doesn't need them)
  - pools across ALL class folders into one unlabeled pool (or keep them
    separate if you want -- see `flatten_classes` below)
  - actually applies blank-tile filtering via `blank_std_threshold`, which
    is accepted as a constructor arg in your original class but never used
    in __getitem__ -- dead code there. Worth filtering here since we're
    writing every tile to disk anyway and don't want to waste storage/
    training compute on blank/near-uniform tiles.
  - writes a manifest.csv mapping each output tile -> source image, so you
    can trace any tile back to its origin later if needed.
"""

import os
import csv
import numpy as np
from PIL import Image
from pathlib import Path


def find_source_images(root_dirs, max_width=8000, threshold=45,
                        min_filesize_bytes=2000, extensions=(".png",)):
    """Same file-discovery logic as MemoryEfficientImageFolder.__init__,
    minus the class-label bookkeeping."""
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    all_samples = []  # (img_path, width, height)
    for current_root_dir in root_dirs:
        if not os.path.isdir(current_root_dir):
            print(f"Skipping {current_root_dir}: not a valid directory.")
            continue

        # Walk every subfolder (class folders or otherwise) -- SSL doesn't
        # care about the class structure, just wants every image found.
        for dirpath, _, filenames in os.walk(current_root_dir):
            for fname in sorted(filenames):
                if not fname.lower().endswith(extensions):
                    continue
                img_path = os.path.join(dirpath, fname)
                if os.path.getsize(img_path) <= min_filesize_bytes:
                    continue
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except Exception:
                    continue
                if width < threshold:
                    continue
                if max_width is not None and width > max_width:
                    continue
                all_samples.append((img_path, width, height))

    return all_samples


def export_tiles(
    root_dirs,
    output_dir,
    tile_size=272,
    tile_step=90,
    resize_factor=3,
    max_width=8000,
    threshold=45,
    min_filesize_bytes=2000,
    blank_std_threshold=25,
    jpeg_quality=95,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = find_source_images(root_dirs, max_width, threshold, min_filesize_bytes)
    print(f"Found {len(samples)} source images")

    manifest_path = output_dir / "manifest.csv"
    n_written, n_skipped_blank = 0, 0

    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["tile_filename", "source_image", "tile_idx", "left", "top"])

        for img_idx, (img_path, orig_w, orig_h) in enumerate(samples):
            stem = Path(img_path).stem
            try:
                with Image.open(img_path) as image:
                    image = image.convert("RGB")
                    new_w, new_h = orig_w * resize_factor, orig_h * resize_factor
                    image = image.resize((new_w, new_h), resample=Image.BICUBIC)

                    n_tiles = (new_w - tile_size) // tile_step + 1
                    for tile_idx in range(n_tiles):
                        left = tile_idx * tile_step
                        if left + tile_size > new_w:
                            left = new_w - tile_size
                        top = 0
                        tile = image.crop((left, top, left + tile_size, top + tile_size))

                        # Blank-tile filter (defined but unused in the
                        # original class) -- actually applied here.
                        tile_arr = np.asarray(tile.convert("L"), dtype=np.float32)
                        if tile_arr.std() < blank_std_threshold:
                            n_skipped_blank += 1
                            continue

                        out_name = f"{stem}_tile{tile_idx:04d}.jpg"
                        tile.save(output_dir / out_name, quality=jpeg_quality)
                        writer.writerow([out_name, img_path, tile_idx, left, top])
                        n_written += 1
            except Exception as e:
                print(f"  Skipping {img_path}: {e}")
                continue

            if img_idx % 50 == 0:
                print(f"  [{img_idx}/{len(samples)}] images processed, "
                      f"{n_written} tiles written so far")

    print(f"\nDone. {n_written} tiles written to {output_dir}")
    print(f"{n_skipped_blank} blank/low-variance tiles filtered out")
    print(f"Manifest: {manifest_path}")
    return output_dir, manifest_path


if __name__ == "__main__":
    import tempfile

    # Smoke test with synthetic images
    with tempfile.TemporaryDirectory() as tmp:
        src_dir = Path(tmp) / "src" / "class_a"
        src_dir.mkdir(parents=True)
        out_dir = Path(tmp) / "out"

        rng = np.random.RandomState(0)
        # one noisy (non-blank) image
        arr = rng.randint(0, 255, (100, 400, 3), dtype=np.uint8)
        Image.fromarray(arr).save(src_dir / "img_001.png")
        # one blank image -- should get filtered out entirely
        blank = np.full((100, 400, 3), 128, dtype=np.uint8)
        Image.fromarray(blank).save(src_dir / "img_002.png")

        out_path, manifest = export_tiles(
            str(Path(tmp) / "src"), out_dir,
            tile_size=64, tile_step=32, threshold=10, min_filesize_bytes=10,
        )
        n_files = len(list(out_path.glob("*.jpg")))
        print(f"\n[TEST] {n_files} tile files on disk, manifest exists: {manifest.exists()}")
        assert n_files > 0, "expected some tiles from the noisy image"
        print("[OK] smoke test passed")
