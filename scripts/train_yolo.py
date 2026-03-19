#!/usr/bin/env python3
"""Train a YOLO object detector on the prepared grocery shelf dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO baseline detector.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("yolo_dataset/data.yaml"),
        help="Path to the YOLO dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Model checkpoint (.pt) or architecture config (.yaml).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional pretrained weights to load when --model is a custom .yaml file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Training image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size. Use -1 for automatic batch sizing.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, for example cpu, 0, or 0,1.",
    )
    parser.add_argument(
        "--project",
        default="runs/detect",
        help="Output project directory.",
    )
    parser.add_argument(
        "--name",
        default="grocery_baseline",
        help="Run name.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {args.data}. Run scripts/prepare_yolo_dataset.py first."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Create a virtual environment and run:\n"
            "pip install -r requirements.txt"
        ) from exc

    model = YOLO(args.model)
    if args.weights is not None:
        model = model.load(args.weights)

    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    print("Training finished.")
    print(results)


if __name__ == "__main__":
    main()
