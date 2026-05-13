"""
auto_label.py
=============
Runs the existing YOLOv8 model over every image in training_captures/
and writes a matching YOLO-format .txt label file next to each image.

After this you can upload the whole training_captures/ folder to Roboflow
and the boxes will already be drawn — just delete / fix the bad ones.

Usage (run on Windows in the project folder):
    python auto_label.py

Optional args:
    python auto_label.py --input training_captures --model models/200923_best_yolov8n.pt --conf 0.3
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT = 'training_captures'
DEFAULT_MODEL = 'models/200923_best_yolov8n.pt'
DEFAULT_CONF  = 0.30   # lower than aimbot so we catch more, review the iffy ones


def label_images(input_dir: str, model_path: str, conf: float):
    input_path = Path(input_dir)
    images = sorted(input_path.glob('*.png')) + sorted(input_path.glob('*.jpg')) + sorted(input_path.glob('*.jpeg'))

    if not images:
        print(f'No images found in "{input_dir}". Run collect_training_data.py first.')
        return

    print(f'Loading model: {model_path}')
    model = YOLO(model_path)

    labelled = 0
    skipped  = 0

    for img_path in images:
        txt_path = img_path.with_suffix('.txt')

        # Skip already-labelled images
        if txt_path.exists():
            skipped += 1
            continue

        results = model(str(img_path), conf=conf, verbose=False)
        boxes   = results[0].boxes

        lines = []
        for box in boxes:
            cls  = int(box.cls[0])
            xywhn = box.xywhn[0].tolist()   # normalised cx, cy, w, h
            lines.append(f'{cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}')

        txt_path.write_text('\n'.join(lines))
        labelled += 1
        det_count = len(lines)
        print(f'  {img_path.name}  →  {det_count} detection(s)')

    print(f'\nDone. {labelled} images labelled, {skipped} already had labels.')
    print(f'Upload the "{input_dir}" folder to Roboflow — boxes will be pre-filled.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help='Folder containing captured PNG screenshots')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='Path to YOLOv8 .pt model')
    parser.add_argument('--conf',  type=float, default=DEFAULT_CONF,
                        help='Detection confidence threshold (0.0 – 1.0)')
    args = parser.parse_args()

    label_images(args.input, args.model, args.conf)


if __name__ == '__main__':
    main()
