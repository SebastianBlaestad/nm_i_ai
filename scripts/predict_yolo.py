#!/usr/bin/env python3
"""Run YOLO inference on shelf images."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO predictions.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a trained YOLO weights file, usually best.pt.",
    )
    parser.add_argument(
        "--source",
        default="train/images",
        help="Image path, folder, or glob pattern to predict on.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size.",
    )
    parser.add_argument(
        "--project",
        default="runs/predict",
        help="Output directory.",
    )
    parser.add_argument(
        "--name",
        default="grocery_predictions",
        help="Prediction run name.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, for example cpu or 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Create a virtual environment and run:\n"
            "pip install -r requirements.txt"
        ) from exc

    model = YOLO(str(args.model))
    predict_kwargs = {
        "source": args.source,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "project": args.project,
        "name": args.name,
        "save": True,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)
    print(f"Prediction finished for {len(results)} batches.")


if __name__ == "__main__":
    main()
