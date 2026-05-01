#!/usr/bin/env python3
"""Train YOLOv11n on the detect_project dataset."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data = script_dir.parent / "yolo_dataset_aug400" / "dataset.yaml"

    parser = argparse.ArgumentParser(description="Train YOLOv11n on detect_project")
    parser.add_argument("--data", type=Path, default=default_data, help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Model checkpoint")
    parser.add_argument("--epochs", type=int, default=240, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=1792, help="Image size")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="CUDA id, cpu, or empty for auto")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers (Windows default: 0)")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project dir")
    parser.add_argument("--name", type=str, default="pv_yolov11n", help="Run name")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer name, e.g. AdamW/SGD")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR factor")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=float, default=5.0, help="Warmup epochs")
    parser.add_argument("--close-mosaic", type=int, default=20, help="Disable mosaic in last N epochs")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience (epochs without improvement). Use 0 to disable.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP (mixed precision). Default is disabled for stability")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    args = parser.parse_args()

    if args.workers is None:
        args.workers = 0 if os.name == "nt" else 8

    return args


def resolve_data_yaml(data_yaml: Path) -> tuple[Path, Path]:
    data_yaml = data_yaml.resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    data_dict = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(data_dict, dict):
        raise ValueError(f"Invalid YAML format: {data_yaml}")

    raw_path = str(data_dict.get("path", ".")).strip()
    root = (data_yaml.parent / raw_path).resolve() if raw_path else data_yaml.parent.resolve()
    data_dict["path"] = str(root)

    for split in ("train", "val"):
        if split not in data_dict or not str(data_dict[split]).strip():
            raise ValueError(f"dataset.yaml must define '{split}'")

    test_rel = str(data_dict.get("test", "")).strip()
    if test_rel:
        data_dict["test"] = test_rel

    tmp = tempfile.NamedTemporaryFile(prefix="yolo11n_data_", suffix=".yaml", delete=False, mode="w", encoding="utf-8")
    with tmp:
        yaml.safe_dump(data_dict, tmp, sort_keys=False, allow_unicode=False)

    return Path(tmp.name), root


def precheck_dataset(dataset_root: Path, data_yaml: Path) -> None:
    data_dict: dict[str, Any] = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    required = [
        dataset_root / str(data_dict["train"]),
        dataset_root / str(data_dict["val"]),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset split folders:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    resolved_yaml, dataset_root = resolve_data_yaml(args.data)
    precheck_dataset(dataset_root, args.data.resolve())

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs: dict[str, Any] = {
        "data": str(resolved_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "rect": True,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "close_mosaic": args.close_mosaic,
        "patience": args.patience,
        "amp": args.amp,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
    }
    if args.device:
        train_kwargs["device"] = args.device

    print(f"Dataset root: {dataset_root}")
    print(f"Resolved data yaml: {resolved_yaml}")
    print(f"Workers: {args.workers}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    mp.freeze_support()
    main()

