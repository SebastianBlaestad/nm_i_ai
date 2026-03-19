# Grocery Shelf Product Detection Baseline

This repository now contains a beginner-friendly baseline for the task:

`Detect products on grocery shelves. Upload your model and score with mAP@0.5.`

The goal of this baseline is not to produce the best score immediately. The goal is to get you from raw data to:

1. understanding what the data is,
2. preparing it in the right format,
3. training a first model,
4. generating predictions you can inspect.

## What You Have

There are two relevant folders in this project:

- `train/`
  Contains shelf photos and `annotations.json`.
- `NM_NGD_product_images/`
  Contains clean reference photos of products, grouped by product id.

For the first version of this project, you should treat `train/` as the main supervised detection dataset.
The reference product images may become useful later for class analysis, retrieval, or zero-shot ideas, but they are not needed to train the first baseline detector.

## Dataset Summary

From the provided labels:

- 248 shelf images
- 22,731 annotated bounding boxes
- 356 product classes

This is a fairly difficult dataset because there are many classes compared with the number of training images. That means your first priority should be a reliable pipeline, not perfect accuracy.

## Recommended First Approach

Use a pretrained YOLO detector and fine-tune it on your dataset.

Why:

- it is widely used for object detection,
- it is much easier for a first project than building a detector from scratch,
- it supports transfer learning, which matters because your dataset is not huge.

This repo includes scripts for:

- converting the COCO annotations to YOLO format,
- splitting images into train and validation sets,
- generating a YOLO dataset config file,
- training a baseline model,
- running inference on images,
- experimenting with a custom YOLO architecture.

## Project Files Added

- `requirements.txt`
- `scripts/prepare_yolo_dataset.py`
- `scripts/train_yolo.py`
- `scripts/predict_yolo.py`
- `models/grocery_yolov8_custom.yaml`
- `models/grocery_yolov8_deeper.yaml`

## Recommended Environment

Use Python `3.10`, `3.11`, or `3.12`.

Your current local Python is newer than what many ML packages usually support first. If `ultralytics` fails to install with your default Python, create a virtual environment with Python `3.11`.

Example:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your machine only has `python3`, try:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Step 1: Prepare the Dataset

Convert the provided COCO annotations into YOLO labels and create a train/validation split:

```bash
python scripts/prepare_yolo_dataset.py
```

This creates:

- `yolo_dataset/images/train`
- `yolo_dataset/images/val`
- `yolo_dataset/labels/train`
- `yolo_dataset/labels/val`
- `yolo_dataset/data.yaml`
- `yolo_dataset/category_mapping.csv`
- `yolo_dataset/split_summary.json`

The split logic tries to avoid putting extremely rare classes only into validation.

## Step 2: Train a Baseline Model

Start with a small pretrained YOLO model:

```bash
python scripts/train_yolo.py --model yolo11n.pt --epochs 50 --imgsz 1280
```

If `yolo11n.pt` is not available in your installed `ultralytics` version, try:

```bash
python scripts/train_yolo.py --model yolov8n.pt --epochs 50 --imgsz 1280
```

A good beginner baseline is:

- model: `yolo11n.pt` or `yolov8n.pt`
- image size: `1280`
- epochs: `50`
- batch: auto

If you have a stronger GPU, you can later try:

- `yolo11s.pt`
- more epochs
- larger image size

## Custom Model Architecture

If you want to edit the network yourself, use:

- `models/grocery_yolov8_custom.yaml`
- `models/grocery_yolov8_deeper.yaml`

The most important concept is:

- the second number on each layer row is the repeat count,
- increasing that repeat count adds more internal blocks,
- in practice, that is the simplest way to add more "hidden layers".

Examples:

- `[-1, 3, C2f, [128, True]]` means that block is repeated `3` times
- changing it to `[-1, 5, C2f, [128, True]]` makes that stage deeper

Recommended first custom run:

```bash
python scripts/train_yolo.py \
  --model models/grocery_yolov8_custom.yaml \
  --weights yolov8n.pt \
  --epochs 50 \
  --imgsz 960
```

Recommended deeper experiment:

```bash
python scripts/train_yolo.py \
  --model models/grocery_yolov8_deeper.yaml \
  --weights yolov8n.pt \
  --epochs 60 \
  --imgsz 960
```

Notes:

- `--model ...yaml` defines the architecture
- `--weights yolov8n.pt` loads pretrained weights into matching layers
- if partial weight loading fails in your installed version, fall back to the standard `.pt` baseline

## Step 3: Run Inference

After training, run predictions on a few shelf images:

```bash
python scripts/predict_yolo.py \
  --model runs/detect/grocery_baseline/weights/best.pt \
  --source train/images \
  --conf 0.25
```

Predicted images will be saved under `runs/predict/...`.

## Training Tips

- Start small and make sure the whole pipeline works before trying to optimize.
- Always inspect a few labels and predictions manually.
- Because there are many classes, class confusion is expected.
- A larger image size often helps shelf detection because products are small.
- If training is unstable or too slow, reduce `imgsz` to `960`.

## What To Try After the Baseline

Once the first model works, the next improvements to try are:

1. Train longer.
2. Try a larger model (`s` instead of `n`).
3. Increase image size.
4. Check whether some labels are noisy or inconsistent.
5. Use test-time augmentation if the competition rules allow it.
6. Explore whether `NM_NGD_product_images/` can help with product recognition or class verification.

## Common Beginner Questions

### What is mAP@0.5?

It is a detection metric. A prediction counts as correct when:

- the predicted class is correct, and
- the predicted box overlaps the ground-truth box enough.

`@0.5` means the IoU threshold is `0.5`.

### What is IoU?

IoU means Intersection over Union. It measures how much the predicted box overlaps the true box.

### Why do we need a validation split?

Because you need a held-out set to estimate whether the model is learning something useful rather than only memorizing training images.

### Which activation function is used?

For the current setup, the architecture follows the standard Ultralytics YOLO blocks, which use `SiLU` in the standard Conv blocks.

### How many epochs should I use?

Use these as practical starting points:

- `30` epochs for a quick smoke test
- `50` epochs for a first real baseline
- `60-100` epochs for custom-architecture experiments if training remains stable

## Suggested First Run

If you want the shortest possible path:

```bash
python scripts/prepare_yolo_dataset.py
python scripts/train_yolo.py --model yolo11n.pt --epochs 30 --imgsz 960
```

Then inspect:

- training plots under `runs/detect/...`
- sample predictions
- validation mAP

## Important Limitation

This repository currently prepares and trains on the provided training set only.

If your assignment platform expects a final uploaded model or predictions for a separate hidden test set, this baseline gets you to the point where you can train a model and produce detections, but the exact submission format will depend on the platform instructions.
