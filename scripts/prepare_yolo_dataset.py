#!/usr/bin/env python3
"""Convert the provided COCO-style shelf dataset into YOLO format."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a YOLO dataset from train/annotations.json."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("train/annotations.json"),
        help="Path to the COCO annotations file.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("train/images"),
        help="Directory containing the shelf images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("yolo_dataset"),
        help="Output directory for YOLO-formatted data.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of images to place in validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    return parser.parse_args()


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_class_mapping(categories: list[dict]) -> tuple[dict[int, int], list[str]]:
    sorted_categories = sorted(categories, key=lambda item: item["id"])
    class_id_to_index = {
        category["id"]: index for index, category in enumerate(sorted_categories)
    }
    class_names = []
    for category in sorted_categories:
        name = str(category.get("name", "")).strip()
        if not name:
            name = f"category_{category['id']}"
        class_names.append(name)
    return class_id_to_index, class_names


def normalize_bbox(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / width
    y_center = (y + h / 2.0) / height
    return x_center, y_center, w / width, h / height


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)


def choose_validation_images(
    image_ids: list[int],
    image_to_classes: dict[int, set[int]],
    val_count: int,
    seed: int,
) -> set[int]:
    rng = random.Random(seed)
    class_image_frequency = Counter()
    for classes in image_to_classes.values():
        class_image_frequency.update(classes)

    protected_images = {
        image_id
        for image_id, classes in image_to_classes.items()
        if any(class_image_frequency[class_id] <= 1 for class_id in classes)
    }

    candidates = [image_id for image_id in image_ids if image_id not in protected_images]
    rng.shuffle(candidates)

    remaining_train_frequency = class_image_frequency.copy()
    validation_ids: set[int] = set()

    for image_id in candidates:
        if len(validation_ids) >= val_count:
            break

        classes = image_to_classes[image_id]
        if any(remaining_train_frequency[class_id] <= 1 for class_id in classes):
            continue

        validation_ids.add(image_id)
        for class_id in classes:
            remaining_train_frequency[class_id] -= 1

    if len(validation_ids) < val_count:
        leftovers = [image_id for image_id in candidates if image_id not in validation_ids]
        for image_id in leftovers:
            if len(validation_ids) >= val_count:
                break
            validation_ids.add(image_id)

    return validation_ids


def write_label_file(
    image: dict,
    annotations: list[dict],
    class_id_to_index: dict[int, int],
    output_path: Path,
) -> int:
    width = image["width"]
    height = image["height"]
    lines = []

    for annotation in annotations:
        bbox = annotation["bbox"]
        x_center, y_center, box_width, box_height = normalize_bbox(bbox, width, height)

        if box_width <= 0 or box_height <= 0:
            continue

        class_index = class_id_to_index[annotation["category_id"]]
        lines.append(
            f"{class_index} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def link_or_copy_image(source: Path, target: Path, copy_images: bool) -> None:
    if copy_images:
        shutil.copy2(source, target)
        return

    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def write_data_yaml(output_dir: Path, class_names: list[str]) -> None:
    names_json = json.dumps(class_names, ensure_ascii=False)
    yaml_text = (
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {names_json}\n"
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def write_category_mapping(
    output_dir: Path,
    categories: list[dict],
    class_id_to_index: dict[int, int],
) -> None:
    mapping_path = output_dir / "category_mapping.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_category_id", "yolo_class_index", "name"])
        for category in sorted(categories, key=lambda item: item["id"]):
            name = str(category.get("name", "")).strip() or f"category_{category['id']}"
            writer.writerow(
                [category["id"], class_id_to_index[category["id"]], name]
            )


def main() -> None:
    args = parse_args()
    coco = load_coco(args.annotations)

    images = coco["images"]
    categories = coco["categories"]
    annotations = coco["annotations"]

    image_by_id = {image["id"]: image for image in images}
    annotations_by_image = defaultdict(list)
    image_to_classes: dict[int, set[int]] = defaultdict(set)

    for annotation in annotations:
        image_id = annotation["image_id"]
        annotations_by_image[image_id].append(annotation)
        image_to_classes[image_id].add(annotation["category_id"])

    class_id_to_index, class_names = build_class_mapping(categories)
    image_ids = sorted(image_by_id)
    val_count = max(1, int(round(len(image_ids) * args.val_fraction)))
    validation_ids = choose_validation_images(
        image_ids=image_ids,
        image_to_classes=image_to_classes,
        val_count=val_count,
        seed=args.seed,
    )

    clean_output_dir(args.output_dir)
    write_data_yaml(args.output_dir, class_names)
    write_category_mapping(args.output_dir, categories, class_id_to_index)

    split_counts = {"train": 0, "val": 0}
    label_counts = {"train": 0, "val": 0}

    for image_id in image_ids:
        image = image_by_id[image_id]
        split = "val" if image_id in validation_ids else "train"
        split_counts[split] += 1

        image_name = image["file_name"]
        source_image_path = args.images_dir / image_name
        target_image_path = args.output_dir / "images" / split / image_name
        target_label_path = (
            args.output_dir / "labels" / split / f"{Path(image_name).stem}.txt"
        )

        if not source_image_path.exists():
            raise FileNotFoundError(f"Missing image file: {source_image_path}")

        link_or_copy_image(source_image_path, target_image_path, args.copy_images)
        label_count = write_label_file(
            image=image,
            annotations=annotations_by_image[image_id],
            class_id_to_index=class_id_to_index,
            output_path=target_label_path,
        )
        label_counts[split] += label_count

    summary = {
        "images_total": len(images),
        "annotations_total": len(annotations),
        "classes_total": len(categories),
        "train_images": split_counts["train"],
        "val_images": split_counts["val"],
        "train_annotations": label_counts["train"],
        "val_annotations": label_counts["val"],
        "output_dir": str(args.output_dir.resolve()),
    }

    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"YOLO dataset prepared at: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
