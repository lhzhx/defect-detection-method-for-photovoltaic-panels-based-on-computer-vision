#!/usr/bin/env python3
"""Validate YOLO dataset structure and label ranges."""

from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Dataset root")
    parser.add_argument("--num-classes", type=int, default=1)
    return parser.parse_args()


def check_split(dataset_dir: Path, split: str, num_classes: int) -> list[str]:
    errors: list[str] = []
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    if not images_dir.exists() or not labels_dir.exists():
        errors.append(f"Missing split dirs for {split}")
        return errors

    images = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    labels = sorted(labels_dir.glob("*.txt"))

    for ip in images:
        lp = labels_dir / f"{ip.stem}.txt"
        if not lp.exists():
            errors.append(f"Missing label: {lp}")

    for lp in labels:
        if not any((images_dir / f"{lp.stem}{ext}").exists() for ext in IMAGE_EXTS):
            errors.append(f"Missing image for label: {lp}")
            continue

        for ln, line in enumerate(lp.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{lp}:{ln} should have 5 columns")
                continue
            try:
                cls = int(parts[0])
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                errors.append(f"{lp}:{ln} non-numeric values")
                continue
            if cls < 0 or cls >= num_classes:
                errors.append(f"{lp}:{ln} class out of range: {cls}")
            if any(v < 0.0 or v > 1.0 for v in vals):
                errors.append(f"{lp}:{ln} coords out of [0,1]")

    return errors


def main() -> None:
    args = parse_args()
    ds = args.dataset_dir.resolve()
    if not ds.exists():
        raise FileNotFoundError(f"Dataset dir not found: {ds}")

    errors: list[str] = []
    for split in ("train", "val", "test"):
        if (ds / "images" / split).exists() or (ds / "labels" / split).exists():
            errors.extend(check_split(ds, split, args.num_classes))

    yaml_path = ds / "dataset.yaml"
    if not yaml_path.exists():
        errors.append(f"Missing dataset.yaml: {yaml_path}")

    if errors:
        print("Validation failed:")
        for e in errors[:50]:
            print(f"  - {e}")
        raise SystemExit(1)

    print("Validation passed.")


if __name__ == "__main__":
    main()

