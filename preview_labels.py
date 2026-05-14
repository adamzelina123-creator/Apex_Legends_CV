"""
preview_labels.py
=================
Draws the auto-generated bounding boxes on your captured screenshots
and saves previews to label_preview/ so you can check they look correct.

Usage:
    python preview_labels.py

Open the label_preview/ folder in Windows Explorer when done.
If boxes look wrong (on walls, items, etc.) you have too few good captures —
go capture more in the firing range and re-run train_pipeline.py.
"""

import os
from pathlib import Path
import cv2
import numpy as np

CAPTURES_DIR  = Path('training_captures')
PREVIEW_DIR   = Path('label_preview')
CLASS_NAMES   = ['avatar']
COLOUR        = (0, 255, 0)   # green boxes


def draw_labels(img_path: Path, txt_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    if txt_path.exists():
        for line in txt_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, bw, bh = int(parts[0]), float(parts[1]), \
                                      float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOUR, 2)
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            cv2.putText(img, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR, 1)
    else:
        cv2.putText(img, 'NO LABEL FILE', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img


def main():
    images = sorted(CAPTURES_DIR.glob('*.jpg')) + \
             sorted(CAPTURES_DIR.glob('*.jpeg')) + \
             sorted(CAPTURES_DIR.glob('*.png'))

    if not images:
        print(f'No images found in "{CAPTURES_DIR}". Run collect_training_data.py first.')
        return

    PREVIEW_DIR.mkdir(exist_ok=True)
    saved = 0

    print(f'Found {len(images)} images. Processing …')
    for i, img_path in enumerate(images, 1):
        txt_path = img_path.with_suffix('.txt')
        preview  = draw_labels(img_path, txt_path)
        if preview is None:
            continue
        out_path = PREVIEW_DIR / img_path.name
        cv2.imwrite(str(out_path), preview)
        saved += 1
        if i % 20 == 0 or i == len(images):
            print(f'  {i}/{len(images)} done …')

    print(f'Saved {saved} preview images to "{PREVIEW_DIR.resolve()}"')
    print('Open that folder in Windows Explorer to check the green boxes.')
    print()
    print('GREEN BOX on a legend  = good')
    print('GREEN BOX on a wall/item = bad (not enough training variety)')
    print('NO BOX on a visible legend = model missed it (capture more angles)')


if __name__ == '__main__':
    main()
