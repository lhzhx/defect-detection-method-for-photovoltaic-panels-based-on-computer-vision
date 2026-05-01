#!/usr/bin/env python3
"""Build a new YOLO dataset from 40 LabelMe JSON samples and expand to 400 images.

Augmentations:
- small-angle rotation
- horizontal flip
- scale + translation
- Gaussian noise
- HSV jitter
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    stem: str
    image_path: Path
    boxes_xyxy: list[tuple[float, float, float, float]]


@dataclass
class AugConfig:
    angle_deg: float
    scale_min: float
    scale_max: float
    translate_frac: float
    fliplr_prob: float
    hsv_h: float
    hsv_s: float
    hsv_v: float
    noise_std_max: float


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build augmented YOLO dataset from images+json")
    parser.add_argument("--images-dir", type=Path, default=root / "images", help="Source image directory")
    parser.add_argument("--annotations-dir", type=Path, default=root / "annotations", help="Source LabelMe JSON directory")
    parser.add_argument("--output-dir", type=Path, default=root / "yolo_dataset_aug400", help="Output YOLO dataset directory")
    parser.add_argument("--target-total", type=int, default=400, help="Target total image count after expansion")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio by source image groups")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--class-name", type=str, default="NG", help="Class name written to dataset.yaml")

    parser.add_argument("--angle", type=float, default=8.0, help="Max absolute rotation angle in degrees")
    parser.add_argument("--scale-min", type=float, default=0.92, help="Min scale factor")
    parser.add_argument("--scale-max", type=float, default=1.08, help="Max scale factor")
    parser.add_argument("--translate", type=float, default=0.08, help="Max translation fraction of image size")
    parser.add_argument("--fliplr-prob", type=float, default=0.5, help="Probability of horizontal flip")
    parser.add_argument("--hsv-h", type=float, default=0.015, help="Hue jitter ratio")
    parser.add_argument("--hsv-s", type=float, default=0.30, help="Saturation jitter ratio")
    parser.add_argument("--hsv-v", type=float, default=0.20, help="Value jitter ratio")
    parser.add_argument("--noise-std", type=float, default=6.0, help="Max Gaussian noise std")

    parser.add_argument("--clean", action="store_true", help="Delete output dir before generation")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned counts")
    return parser.parse_args()


def read_image(path: Path) -> Optional[np.ndarray]:
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix.lower(), image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    buf.tofile(str(path))


def load_labelme_rectangles(json_path: Path) -> tuple[str, list[tuple[float, float, float, float]]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_rel = Path(str(data.get("imagePath", "")).replace("\\", "/")).name
    shapes = data.get("shapes", [])

    boxes: list[tuple[float, float, float, float]] = []
    for s in shapes:
        if s.get("shape_type") != "rectangle":
            continue
        points = s.get("points", [])
        if len(points) != 2:
            continue
        (x1, y1), (x2, y2) = points
        xa, xb = sorted([float(x1), float(x2)])
        ya, yb = sorted([float(y1), float(y2)])
        if xb - xa < 1.0 or yb - ya < 1.0:
            continue
        boxes.append((xa, ya, xb, yb))

    return image_rel, boxes


def find_samples(images_dir: Path, annotations_dir: Path) -> list[Sample]:
    samples: list[Sample] = []

    json_files = sorted(annotations_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {annotations_dir}")

    for jp in json_files:
        image_name, boxes = load_labelme_rectangles(jp)
        if not image_name:
            continue
        ip = images_dir / image_name
        if not ip.exists():
            continue
        if ip.suffix.lower() not in IMAGE_EXTS:
            continue
        if not boxes:
            continue
        samples.append(Sample(stem=ip.stem, image_path=ip, boxes_xyxy=boxes))

    if not samples:
        raise RuntimeError("No valid image+rectangle annotation pairs found.")
    return samples


def apply_hsv_jitter(img: np.ndarray, rng: random.Random, cfg: AugConfig) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_shift = rng.uniform(-cfg.hsv_h, cfg.hsv_h) * 179.0
    s_gain = 1.0 + rng.uniform(-cfg.hsv_s, cfg.hsv_s)
    v_gain = 1.0 + rng.uniform(-cfg.hsv_v, cfg.hsv_v)

    hsv[..., 0] = (hsv[..., 0] + h_shift) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * s_gain, 0.0, 255.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * v_gain, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_noise(img: np.ndarray, rng: random.Random, cfg: AugConfig) -> np.ndarray:
    sigma = rng.uniform(0.0, cfg.noise_std_max)
    if sigma <= 0.0:
        return img
    noise = np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0.0, 255.0)
    return out.astype(np.uint8)


def affine_transform(
    image: np.ndarray,
    boxes_xyxy: list[tuple[float, float, float, float]],
    rng: random.Random,
    cfg: AugConfig,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    h, w = image.shape[:2]

    angle = rng.uniform(-cfg.angle_deg, cfg.angle_deg)
    scale = rng.uniform(cfg.scale_min, cfg.scale_max)
    tx = rng.uniform(-cfg.translate_frac, cfg.translate_frac) * w
    ty = rng.uniform(-cfg.translate_frac, cfg.translate_frac) * h

    center = (w * 0.5, h * 0.5)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty

    out = cv2.warpAffine(image, m, dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    out_boxes: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in boxes_xyxy:
        corners = np.array(
            [
                [x1, y1, 1.0],
                [x2, y1, 1.0],
                [x2, y2, 1.0],
                [x1, y2, 1.0],
            ],
            dtype=np.float32,
        )
        warped = (m @ corners.T).T
        nx1 = float(np.clip(np.min(warped[:, 0]), 0, w - 1))
        ny1 = float(np.clip(np.min(warped[:, 1]), 0, h - 1))
        nx2 = float(np.clip(np.max(warped[:, 0]), 0, w - 1))
        ny2 = float(np.clip(np.max(warped[:, 1]), 0, h - 1))

        if nx2 - nx1 >= 2.0 and ny2 - ny1 >= 2.0:
            out_boxes.append((nx1, ny1, nx2, ny2))

    return out, out_boxes


def maybe_fliplr(
    image: np.ndarray,
    boxes_xyxy: list[tuple[float, float, float, float]],
    rng: random.Random,
    fliplr_prob: float,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    if rng.random() >= fliplr_prob:
        return image, boxes_xyxy

    h, w = image.shape[:2]
    out = cv2.flip(image, 1)
    out_boxes: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in boxes_xyxy:
        nx1 = (w - 1) - x2
        nx2 = (w - 1) - x1
        out_boxes.append((nx1, y1, nx2, y2))
    return out, out_boxes


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[float, float, float, float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw * 0.5
    cy = y1 + bh * 0.5
    return (cx / w, cy / h, bw / w, bh / h)


def save_yolo_label(path: Path, boxes_xyxy: list[tuple[float, float, float, float]], w: int, h: int) -> None:
    lines: list[str] = []
    for x1, y1, x2, y2 in boxes_xyxy:
        cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    txt = "\n".join(lines)
    if txt:
        txt += "\n"
    path.write_text(txt, encoding="utf-8")


def ensure_output_layout(output_dir: Path, clean: bool) -> None:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    for p in [
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def build_dataset_yaml(output_dir: Path, class_name: str) -> None:
    data = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    (output_dir / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.target_total < 1:
        raise ValueError("--target-total must be >= 1")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0,1)")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    samples = find_samples(args.images_dir.resolve(), args.annotations_dir.resolve())
    n = len(samples)

    base = args.target_total // n
    rem = args.target_total % n
    if base < 1:
        raise ValueError(f"target_total={args.target_total} is too small for {n} source images")

    samples_sorted = sorted(samples, key=lambda s: s.stem)
    rng.shuffle(samples_sorted)

    variants_per_source: dict[str, int] = {}
    for i, s in enumerate(samples_sorted):
        variants_per_source[s.stem] = base + (1 if i < rem else 0)

    val_source_count = max(1, int(round(n * args.val_ratio)))
    val_source_stems = {s.stem for s in samples_sorted[:val_source_count]}

    train_total = sum(variants_per_source[s.stem] for s in samples if s.stem not in val_source_stems)
    val_total = sum(variants_per_source[s.stem] for s in samples if s.stem in val_source_stems)

    print(f"Source images: {n}")
    print(f"Target total: {args.target_total}")
    print(f"Planned split: train={train_total}, val={val_total}")

    if args.dry_run:
        return

    out_dir = args.output_dir.resolve()
    ensure_output_layout(out_dir, clean=args.clean)

    cfg = AugConfig(
        angle_deg=args.angle,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        translate_frac=args.translate,
        fliplr_prob=args.fliplr_prob,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        noise_std_max=args.noise_std,
    )

    created = 0
    dropped_boxes = 0
    manifest_rows: list[dict[str, Any]] = []

    for sample in sorted(samples, key=lambda s: s.stem):
        image = read_image(sample.image_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {sample.image_path}")
        h, w = image.shape[:2]

        split = "val" if sample.stem in val_source_stems else "train"
        k = variants_per_source[sample.stem]

        for i in range(k):
            out_boxes = list(sample.boxes_xyxy)
            out_img = image.copy()

            if i > 0:
                out_img, out_boxes = affine_transform(out_img, out_boxes, rng, cfg)
                out_img, out_boxes = maybe_fliplr(out_img, out_boxes, rng, cfg.fliplr_prob)
                out_img = apply_hsv_jitter(out_img, rng, cfg)
                out_img = apply_noise(out_img, rng, cfg)

            if not out_boxes:
                dropped_boxes += 1
                out_boxes = list(sample.boxes_xyxy)

            stem = f"{sample.stem}_v{i:02d}"
            out_img_path = out_dir / "images" / split / f"{stem}.jpg"
            out_lbl_path = out_dir / "labels" / split / f"{stem}.txt"

            write_image(out_img_path, out_img)
            save_yolo_label(out_lbl_path, out_boxes, w, h)
            created += 1

            manifest_rows.append(
                {
                    "source": sample.stem,
                    "variant": i,
                    "split": split,
                    "image": str(out_img_path.relative_to(out_dir)).replace("\\", "/"),
                    "label": str(out_lbl_path.relative_to(out_dir)).replace("\\", "/"),
                }
            )

    build_dataset_yaml(out_dir, args.class_name)

    manifest = {
        "source_images": n,
        "target_total": args.target_total,
        "created_total": created,
        "train_total": len([r for r in manifest_rows if r["split"] == "train"]),
        "val_total": len([r for r in manifest_rows if r["split"] == "val"]),
        "dropped_box_recoveries": dropped_boxes,
        "seed": args.seed,
        "augment": {
            "angle": args.angle,
            "scale_min": args.scale_min,
            "scale_max": args.scale_max,
            "translate": args.translate,
            "fliplr_prob": args.fliplr_prob,
            "hsv_h": args.hsv_h,
            "hsv_s": args.hsv_s,
            "hsv_v": args.hsv_v,
            "noise_std": args.noise_std,
        },
        "rows": manifest_rows,
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Created dataset: {out_dir}")
    print(f"Created images: {created}")
    print(f"Dropped-box recoveries: {dropped_boxes}")
    print(f"Wrote: {out_dir / 'dataset.yaml'}")
    print(f"Wrote: {out_dir / 'build_manifest.json'}")


if __name__ == "__main__":
    main()

