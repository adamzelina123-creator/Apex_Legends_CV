"""
train_pipeline.py
=================
Fully automated training pipeline. No Roboflow, no manual labelling.

Steps it runs automatically:
  1. Auto-label all images in training_captures/ using the current model
  2. Split into train / val sets (80 / 20)
  3. Build the YOLO dataset folder structure
  4. Write data.yaml
  5. Fine-tune the model
  6. Copy the best weights into models/ ready to use

Usage (run on Windows in the project folder):
    python train_pipeline.py

Optional args:
    python train_pipeline.py --captures training_captures
                             --base-model models/200923_best_yolov8n.pt
                             --epochs 50
                             --conf 0.30
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Defaults ──────────────────────────────────────────────────────────────────
CAPTURES_DIR = 'training_captures'
BASE_MODEL   = 'models/200923_best_yolov8n.pt'
DATASET_DIR  = 'dataset'
OUTPUT_MODEL = 'models/apex_trained.pt'
EPOCHS       = 50
CONF         = 0.30
VAL_SPLIT    = 0.20
CLASS_NAMES  = ['avatar']


# ── Step 1: Auto-label ────────────────────────────────────────────────────────
def auto_label(captures_dir: Path, model: YOLO, conf: float):
    images = sorted(captures_dir.glob('*.jpg')) + \
             sorted(captures_dir.glob('*.jpeg')) + \
             sorted(captures_dir.glob('*.png'))

    if not images:
        raise FileNotFoundError(
            f'No images found in "{captures_dir}". '
            'Run collect_training_data.py first then come back.'
        )

    print(f'\n[1/4] Auto-labelling {len(images)} images …')
    new_labels = 0
    for img_path in images:
        txt_path = img_path.with_suffix('.txt')
        if txt_path.exists():
            continue                      # already labelled, skip

        results = model(str(img_path), conf=conf, verbose=False)
        lines = []
        for box in results[0].boxes:
            cls   = int(box.cls[0])
            cx, cy, w, h = box.xywhn[0].tolist()
            lines.append(f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')

        txt_path.write_text('\n'.join(lines))
        new_labels += 1

    print(f'    {new_labels} new label files written '
          f'({len(images) - new_labels} already existed).')
    return images


# ── Step 2: Build dataset folder structure ────────────────────────────────────
def build_dataset(images: list, dataset_dir: Path, val_split: float):
    print(f'\n[2/4] Building dataset in "{dataset_dir}" …')

    # Remove old dataset so we start fresh
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for split in ('train', 'val'):
        (dataset_dir / split / 'images').mkdir(parents=True)
        (dataset_dir / split / 'labels').mkdir(parents=True)

    random.shuffle(images)
    val_count = max(1, int(len(images) * val_split))
    splits = {'val': images[:val_count], 'train': images[val_count:]}

    for split_name, split_images in splits.items():
        for img_path in split_images:
            # Copy image
            dst_img = dataset_dir / split_name / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)
            # Copy label (empty file if no detections)
            src_lbl = img_path.with_suffix('.txt')
            dst_lbl = dataset_dir / split_name / 'labels' / img_path.with_suffix('.txt').name
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text('')

    print(f'    train: {len(splits["train"])}  |  val: {len(splits["val"])}')


# ── Step 3: Write data.yaml ───────────────────────────────────────────────────
def write_yaml(dataset_dir: Path, class_names: list) -> Path:
    yaml_path = dataset_dir / 'data.yaml'
    # Use forward slashes / absolute path so YOLO finds it on Windows
    abs_dir = str(dataset_dir.resolve()).replace('\\', '/')
    yaml_content = (
        f'path: {abs_dir}\n'
        f'train: train/images\n'
        f'val:   val/images\n'
        f'nc: {len(class_names)}\n'
        f'names: {class_names}\n'
    )
    yaml_path.write_text(yaml_content)
    print(f'\n[3/4] data.yaml written → {yaml_path}')
    return yaml_path


# ── Step 4: Train ─────────────────────────────────────────────────────────────
def train(base_model_path: str, yaml_path: Path, epochs: int, output_model: str):
    print(f'\n[4/4] Training for {epochs} epochs … (this takes a while)\n')
    model = YOLO(base_model_path)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=8,
        patience=15,          # stop early if no improvement
        plots=False,          # skip plots to save time
        verbose=True,
    )

    # Copy best weights to models/
    best = Path(results.save_dir) / 'weights' / 'best.pt'
    if best.exists():
        Path(output_model).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, output_model)
        print(f'\n✓ New model saved → {output_model}')
        print(f'  Update MODEL_PATH in aimbot_Baseline.py to use it:\n'
              f'  MODEL_PATH = \'{output_model}\'')
    else:
        print(f'Training finished but best.pt not found at {best}')


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captures',   default=CAPTURES_DIR)
    parser.add_argument('--base-model', default=BASE_MODEL)
    parser.add_argument('--epochs',     type=int,   default=EPOCHS)
    parser.add_argument('--conf',       type=float, default=CONF)
    args = parser.parse_args()

    captures_dir = Path(args.captures)
    dataset_dir  = Path(DATASET_DIR)

    print('═' * 55)
    print('  Apex Legends – Auto Training Pipeline')
    print(f'  Captures : {captures_dir}')
    print(f'  Model    : {args.base_model}')
    print(f'  Epochs   : {args.epochs}')
    print('═' * 55)

    model  = YOLO(args.base_model)
    images = auto_label(captures_dir, model, args.conf)
    build_dataset(images, dataset_dir, VAL_SPLIT)
    yaml   = write_yaml(dataset_dir, CLASS_NAMES)
    train(args.base_model, yaml, args.epochs, OUTPUT_MODEL)


if __name__ == '__main__':
    main()
