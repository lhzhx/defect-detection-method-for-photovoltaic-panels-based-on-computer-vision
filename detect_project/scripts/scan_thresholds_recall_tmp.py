#!/usr/bin/env python3
"""Recall-focused threshold scan with optional test-set verification."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def parse_float_grid(text: str, name: str) -> list[float]:
    values: list[float] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"Invalid {name} value: {item}") from exc
        values.append(value)

    if not values:
        raise ValueError(f"{name} grid cannot be empty")
    return values


def parse_int_grid(text: str, name: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid {name} value: {item}") from exc
        values.append(value)

    if not values:
        raise ValueError(f"{name} grid cannot be empty")
    return values


def parse_bool_grid(text: str, name: str) -> list[bool]:
    out: list[bool] = []
    for part in text.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif item in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid {name} value: {part}")
    if not out:
        raise ValueError(f"{name} grid cannot be empty")
    return out


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Recall-focused threshold scan + test verification")
    parser.add_argument(
        "--weights",
        type=Path,
        default=script_dir / "runs" / "detect" / "runs" / "detect" / "pv_yolov11n3" / "weights" / "best.pt",
        help="Model weights",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir.parent / "yolo_dataset_aug400" / "dataset.yaml",
        help="Dataset yaml",
    )
    parser.add_argument("--imgsz", type=int, default=1792)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--val-split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--test-split", type=str, default="test", choices=["train", "val", "test"])

    parser.add_argument("--conf-grid", type=str, default="0.01,0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--iou-grid", type=str, default="0.45,0.55,0.65,0.75")
    parser.add_argument("--max-det-grid", type=str, default="300,600")
    parser.add_argument("--tta-grid", type=str, default="false,true")

    parser.add_argument("--precision-floor", type=float, default=0.40, help="Precision floor for balanced recall pick")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--test-top-k", type=int, default=3, help="Verify top-K val candidates on test split")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=script_dir / "runs" / "eval" / "threshold_scan_recall_plus_test.json",
    )
    return parser.parse_args()


def resolve_data_yaml(data: Path) -> Path:
    data = data.resolve()
    cfg = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML: {data}")
    root = (data.parent / str(cfg.get("path", "."))).resolve()
    cfg["path"] = str(root)
    tmp = tempfile.NamedTemporaryFile(prefix="scan_data_", suffix=".yaml", delete=False, mode="w", encoding="utf-8")
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False, allow_unicode=False)
    return Path(tmp.name)


def extract_metrics(result: Any) -> dict[str, float]:
    box = result.box
    return {
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "mAP50": float(getattr(box, "map50", 0.0)),
        "mAP50_95": float(getattr(box, "map", 0.0)),
    }


def score_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    m = row["metrics"]
    return (float(m["recall"]), float(m["mAP50_95"]), float(m["mAP50"]), float(m["precision"]))


def run_eval(model: YOLO, data_yaml: Path, split: str, args: argparse.Namespace, conf: float, iou: float, max_det: int, tta: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "split": split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "augment": tta,
    }
    if args.device:
        kwargs["device"] = args.device

    result = model.val(**kwargs)
    return {
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "tta": tta,
        "split": split,
        "metrics": extract_metrics(result),
    }


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    resolved_data = resolve_data_yaml(args.data)
    conf_grid = parse_float_grid(args.conf_grid, "conf")
    iou_grid = parse_float_grid(args.iou_grid, "iou")
    max_det_grid = parse_int_grid(args.max_det_grid, "max_det")
    tta_grid = parse_bool_grid(args.tta_grid, "tta")

    model = YOLO(str(weights))

    total = len(conf_grid) * len(iou_grid) * len(max_det_grid) * len(tta_grid)
    print(f"Val scan total: {total}")
    rows: list[dict[str, Any]] = []
    idx = 0

    for conf in conf_grid:
        for iou in iou_grid:
            for max_det in max_det_grid:
                for tta in tta_grid:
                    idx += 1
                    print(
                        f"[{idx}/{total}] split={args.val_split} conf={conf:.2f} iou={iou:.2f} "
                        f"max_det={max_det} tta={tta}"
                    )
                    row = run_eval(model, resolved_data, args.val_split, args, conf, iou, max_det, tta)
                    m = row["metrics"]
                    print(
                        "  -> "
                        f"P={m['precision']:.4f} R={m['recall']:.4f} "
                        f"mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50_95']:.4f}"
                    )
                    rows.append(row)

    ranked = sorted(rows, key=score_key, reverse=True)
    best_recall = ranked[0]
    constrained = [r for r in ranked if r["metrics"]["precision"] >= args.precision_floor]
    best_recall_with_floor = constrained[0] if constrained else None

    print("\nTop candidates (recall-first):")
    for i, row in enumerate(ranked[: args.top_k], start=1):
        m = row["metrics"]
        print(
            f"{i:>2}. conf={row['conf']:.2f} iou={row['iou']:.2f} max_det={row['max_det']:<4} tta={str(row['tta']):<5} "
            f"P={m['precision']:.4f} R={m['recall']:.4f} mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50_95']:.4f}"
        )

    if best_recall_with_floor:
        m = best_recall_with_floor["metrics"]
        print(
            "\nBest with precision floor: "
            f"P={m['precision']:.4f} R={m['recall']:.4f} "
            f"(conf={best_recall_with_floor['conf']:.2f}, iou={best_recall_with_floor['iou']:.2f}, "
            f"max_det={best_recall_with_floor['max_det']}, tta={best_recall_with_floor['tta']})"
        )
    else:
        print("\nNo candidate met precision floor.")

    test_candidates = ranked[: max(1, args.test_top_k)]
    print(f"\nTesting top {len(test_candidates)} val candidates on split='{args.test_split}'...")
    test_rows: list[dict[str, Any]] = []
    for i, row in enumerate(test_candidates, start=1):
        tested = run_eval(
            model,
            resolved_data,
            args.test_split,
            args,
            conf=float(row["conf"]),
            iou=float(row["iou"]),
            max_det=int(row["max_det"]),
            tta=bool(row["tta"]),
        )
        test_rows.append(tested)
        m = tested["metrics"]
        print(
            f"  [{i}] conf={tested['conf']:.2f} iou={tested['iou']:.2f} max_det={tested['max_det']} tta={tested['tta']} "
            f"-> P={m['precision']:.4f} R={m['recall']:.4f} mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50_95']:.4f}"
        )

    ranked_test = sorted(test_rows, key=score_key, reverse=True)
    best_test = ranked_test[0]

    output = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "weights": str(weights),
        "data": str(args.data.resolve()),
        "resolved_data": str(resolved_data),
        "val_split": args.val_split,
        "test_split": args.test_split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "conf_grid": conf_grid,
        "iou_grid": iou_grid,
        "max_det_grid": max_det_grid,
        "tta_grid": tta_grid,
        "precision_floor": args.precision_floor,
        "val_results_ranked": ranked,
        "best_recall_on_val": best_recall,
        "best_recall_with_precision_floor_on_val": best_recall_with_floor,
        "test_validation_ranked": ranked_test,
        "best_on_test_from_top_val_candidates": best_test,
    }

    out_path = args.output_json.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    best_val_metrics = best_recall["metrics"]
    print(
        "\nBest val recall candidate: "
        f"conf={best_recall['conf']:.2f}, iou={best_recall['iou']:.2f}, max_det={best_recall['max_det']}, "
        f"tta={best_recall['tta']}, R={best_val_metrics['recall']:.4f}, P={best_val_metrics['precision']:.4f}"
    )
    best_test_metrics = best_test["metrics"]
    print(
        "Best test candidate (from top val picks): "
        f"conf={best_test['conf']:.2f}, iou={best_test['iou']:.2f}, max_det={best_test['max_det']}, "
        f"tta={best_test['tta']}, R={best_test_metrics['recall']:.4f}, P={best_test_metrics['precision']:.4f}"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

