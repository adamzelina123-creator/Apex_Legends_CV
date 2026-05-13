"""
collect_training_data.py
========================
Captures in-game screenshots for building a legend training dataset.

Controls
--------
  F2       – toggle capture ON / OFF
  F3       – quit

Output
------
  training_captures/YYYYMMDD_HHMMSS_NNNN.png   (one per captured frame)

Capture region mirrors aimbot_Baseline.py exactly:
  CAPTURE_SIZE × CAPTURE_SIZE pixels at (CROP_X, CROP_Y) on the monitor.

After collecting frames, label them with Roboflow or LabelImg and
fine-tune the YOLOv8 model:
  yolo train model=models/200923_best_yolov8n.pt data=data.yaml epochs=50
"""

import os
import time
import datetime
import threading

import mss
import cv2
import numpy as np
from pynput import keyboard

# ── Region must match aimbot_Baseline.py ─────────────────────────────────────
SCREEN_W     = 1600
SCREEN_H     = 900
CAPTURE_SIZE = 900          # square crop side in pixels
CROP_X       = (SCREEN_W - CAPTURE_SIZE) // 2   # 350
CROP_Y       = 0

CAPTURE_FPS  = 2            # frames per second when capturing
OUTPUT_DIR   = 'training_captures'

# ── State ─────────────────────────────────────────────────────────────────────
capturing    = False
running      = True
frame_count  = 0
_lock        = threading.Lock()


def on_press(key):
    global capturing, running
    try:
        if key == keyboard.Key.f2:
            with _lock:
                capturing = not capturing
            state = 'ON' if capturing else 'OFF'
            print(f'[collect] Capture {state}')
        elif key == keyboard.Key.f3:
            running = False
            print('[collect] Stopping …')
            return False          # stops the listener
    except Exception:
        pass


def capture_loop():
    global frame_count
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    region = {
        'left':   CROP_X,
        'top':    CROP_Y,
        'width':  CAPTURE_SIZE,
        'height': CAPTURE_SIZE,
    }
    interval = 1.0 / CAPTURE_FPS

    with mss.mss() as sct:
        while running:
            with _lock:
                should_capture = capturing

            if should_capture:
                frame   = sct.grab(region)
                img     = np.array(frame)[:, :, :3]   # BGR, drop alpha
                ts      = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                fname   = os.path.join(OUTPUT_DIR, f'{ts}_{frame_count:04d}.png')
                cv2.imwrite(fname, img)
                frame_count += 1
                print(f'[collect] Saved {fname}')

            time.sleep(interval)


def main():
    print('═' * 55)
    print('  Apex Legends – Training Data Collector')
    print('  F2  →  toggle capture  |  F3  →  quit')
    print(f'  Output folder: {os.path.abspath(OUTPUT_DIR)}')
    print('═' * 55)

    # Start capture thread
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    # Keyboard listener blocks until F3
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    t.join(timeout=2)
    print(f'[collect] Done. {frame_count} frames saved to "{OUTPUT_DIR}/"')


if __name__ == '__main__':
    main()
