#!/usr/bin/env python3
"""Repartition a YOLO dataset into train/val/test with source-group anti-leakage.

Source grouping rule:
- filename stem like <source>_vXX -> grouped by <source>
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1] / "yolo_dataset_aug400"
    parser = argparse.ArgumentParser(description="Repartition YOLO dataset with grouped anti-leakage")
    parser.add_argument("--dataset-dir", type=Path, default=root, help="YOLO dataset root")
    parser.add_argument("--train-count", type=int, default=320)
    parser.add_argument("--val-count", type=int, default=40)
    parser.add_argument("--test-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-pattern", type=str, default=r"^(.*)_v\d{2}$", help="Regex to extract source id")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_id_from_stem(stem: str, regex: re.Pattern[str]) -> str:
    m = regex.match(stem)
    return m.group(1) if m else stem


def collect_pairs(dataset_dir: Path) -> List[Tuple[str, Path, Path]]:
    pairs: List[Tuple[str, Path, Path]] = []
    for split in ("train", "val", "test"):
        idir = dataset_dir / "images" / split
        ldir = dataset_dir / "labels" / split
        if not idir.exists() or not ldir.exists():
            continue
        for ip in sorted(idir.iterdir()):
            if not ip.is_file() or ip.suffix.lower() not in IMAGE_EXTS:
                continue
            lp = ldir / f"{ip.stem}.txt"
            if not lp.exists():
                raise FileNotFoundError(f"Missing label for image: {ip}")
            pairs.append((ip.stem, ip.resolve(), lp.resolve()))
    if not pairs:
        raise RuntimeError("No image-label pairs found under images/* and labels/*")
    return pairs


def choose_groups_exact(group_sizes: Dict[str, int], target: int, seed: int) -> List[str]:
    # Subset-sum DP to hit exact target while using source groups as atomic units.
    groups = sorted(group_sizes.keys())
    rnd = random.Random(seed)
    rnd.shuffle(groups)

    dp: Dict[int, List[str]] = {0: []}
    for g in groups:
        size = group_sizes[g]
        current = list(dp.items())
        for s, chosen in current:
            ns = s + size
            if ns > target or ns in dp:
                continue
            dp[ns] = chosen + [g]
    if target not in dp:
        raise RuntimeError(f"Cannot reach exact target={target} with source groups")
    return dp[target]


def clear_split(dataset_dir: Path, split: str) -> None:
    for sub in ("images", "labels"):
        d = dataset_dir / sub / split
        d.mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():
            if p.is_file():
                p.unlink()


def update_dataset_yaml(dataset_dir: Path) -> None:
    y = dataset_dir / "dataset.yaml"
    if y.exists():
        data = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
    else:
        data = {}
    if not isinstance(data, dict):
        data = {}

    data.setdefault("path", ".")
    data["train"] = "images/train"
    data["val"] = "images/val"
    data["test"] = "images/test"
    if "names" not in data:
        data["names"] = {0: "NG"}

    y.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    total_target = args.train_count + args.val_count + args.test_count

    pairs = collect_pairs(dataset_dir)
    if len(pairs) != total_target:
        raise RuntimeError(f"Dataset has {len(pairs)} pairs, but target total is {total_target}")

    regex = re.compile(args.group_pattern)
    groups: Dict[str, List[Tuple[str, Path, Path]]] = {}
    for stem, ip, lp in pairs:
        gid = source_id_from_stem(stem, regex)
        groups.setdefault(gid, []).append((stem, ip, lp))

    group_sizes = {g: len(v) for g, v in groups.items()}

    train_groups = choose_groups_exact(group_sizes, args.train_count, seed=args.seed)
    remaining_1 = {g: c for g, c in group_sizes.items() if g not in set(train_groups)}
    val_groups = choose_groups_exact(remaining_1, args.val_count, seed=args.seed + 1)
    test_groups = [g for g in remaining_1.keys() if g not in set(val_groups)]

    test_count = sum(group_sizes[g] for g in test_groups)
    if test_count != args.test_count:
        raise RuntimeError(f"Computed test_count={test_count}, expected {args.test_count}")

    split_map: Dict[str, str] = {}
    for g in train_groups:
        split_map[g] = "train"
    for g in val_groups:
        split_map[g] = "val"
    for g in test_groups:
        split_map[g] = "test"

    print(f"groups: total={len(groups)}, train={len(train_groups)}, val={len(val_groups)}, test={len(test_groups)}")
    print(
        "pairs: "
        f"train={sum(group_sizes[g] for g in train_groups)}, "
        f"val={sum(group_sizes[g] for g in val_groups)}, "
        f"test={sum(group_sizes[g] for g in test_groups)}"
    )

    if args.dry_run:
        return

    snapshot = dataset_dir / "_before_repartition_backup"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            src = dataset_dir / sub / split
            if src.exists():
                dst = snapshot / sub / split
                dst.mkdir(parents=True, exist_ok=True)
                for p in src.iterdir():
                    if p.is_file():
                        shutil.copy2(p, dst / p.name)

    for split in ("train", "val", "test"):
        clear_split(dataset_dir, split)

    for gid, items in groups.items():
        split = split_map[gid]
        for stem, ip, lp in items:
            src_img = ip
            src_lbl = lp
            if not src_img.exists() or not src_lbl.exists():
                # Read from snapshot if source files were cleared before copy.
                src_img = snapshot / "images" / ip.parent.name / ip.name
                src_lbl = snapshot / "labels" / lp.parent.name / lp.name
            if not src_img.exists() or not src_lbl.exists():
                raise FileNotFoundError(f"Missing source pair in live/snapshot: {ip} | {lp}")

            out_i = dataset_dir / "images" / split / ip.name
            out_l = dataset_dir / "labels" / split / f"{stem}.txt"
            shutil.copy2(src_img, out_i)
            shutil.copy2(src_lbl, out_l)

    update_dataset_yaml(dataset_dir)
    print(f"Updated dataset: {dataset_dir}")
    print(f"Backup snapshot: {snapshot}")


if __name__ == "__main__":
    main()

