from time import sleep, perf_counter
import random
from pynput.mouse import Controller as MouseController, Listener as MouseListener, Button
from pynput.keyboard import Listener as KeyboardListener, Key
from ultralytics import YOLO
from mss import mss
import cv2
import numpy as np
import sys
import threading
import torch

# ── Model ──────────────────────────────────────────────────────────────────────
print("//// LOADING MODEL ////")
model = YOLO('models/200923_best_yolov8n.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")
model.to(device)

# ── Screen info ────────────────────────────────────────────────────────────────
_sct        = mss()
_monitor    = _sct.monitors[1]
SCREEN_W    = _monitor['width']
SCREEN_H    = _monitor['height']
# Increase CAPTURE_SIZE for more range (covers wider area, scaled to 640 for model)
# 640 = tight center, 960 = ~50% more range, 1280 = ~100% more range
CAPTURE_SIZE = min(960, SCREEN_W, SCREEN_H)
CROP_X      = (SCREEN_W - CAPTURE_SIZE) // 2
CROP_Y      = (SCREEN_H - CAPTURE_SIZE) // 2

# ── Shared state ───────────────────────────────────────────────────────────────
mouse_controller    = MouseController()
right_click_pressed = False
left_click_pressed  = False
running             = True
aimbot_enabled      = True   # Toggle with F2
latest_boxes        = []   # list of (x1, y1, x2, y2, conf)
boxes_lock          = threading.Lock()

# ── Mouse listener ─────────────────────────────────────────────────────────────
def on_mouse_press(x, y, button, pressed):
    global right_click_pressed, left_click_pressed
    if button == Button.right:
        right_click_pressed = pressed
    elif button == Button.left:
        left_click_pressed = pressed

def mouse_listener_thread():
    with MouseListener(on_click=on_mouse_press) as listener:
        listener.join()

# ── Keyboard listener (F2 = toggle aimbot) ───────────────────────────────────
def on_key_press(key):
    global aimbot_enabled
    if key == Key.f2:
        aimbot_enabled = not aimbot_enabled
        print(f"Aimbot {'ENABLED' if aimbot_enabled else 'DISABLED'}")

def keyboard_listener_thread():
    with KeyboardListener(on_press=on_key_press) as listener:
        listener.join()

# ── Fast screen capture ────────────────────────────────────────────────────────
def get_frame():
    region = {'left': CROP_X, 'top': CROP_Y, 'width': CAPTURE_SIZE, 'height': CAPTURE_SIZE}
    sct_img = _sct.grab(region)
    img = np.frombuffer(sct_img.bgra, dtype=np.uint8).reshape(CAPTURE_SIZE, CAPTURE_SIZE, 4)
    rgb = img[..., :3][..., ::-1].copy()   # BGRA → RGB
    if CAPTURE_SIZE != 640:
        rgb = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_NEAREST)
    return rgb

# ── Detection + aimbot loop ─────────────────────────────────────────────────────
DETECT_FPS_ACTIVE = 20          # FPS when aiming (button held)
DETECT_FPS_IDLE   = 10          # FPS when not aiming (saves CPU/GPU)

def detection_loop(detect_param):
    global latest_boxes, running
    while running:
        t0 = perf_counter()
        frame = get_frame()
        results = model(frame, verbose=False, conf=detect_param, device=device, half=(device == 'cuda'))

        boxes_detected = []
        best_dist  = float('inf')
        best_move  = (0, 0)
        should_move = False

        if len(results[0].boxes):
            for i, box in enumerate(results[0].boxes.xyxy):
                conf     = results[0].boxes.conf[i].item()
                cls_name = model.names[int(results[0].boxes.cls[i])]
                if cls_name != 'avatar':
                    continue
                x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                boxes_detected.append((x1, y1, x2, y2, conf))

                x_center = (x1 + x2) / 2
                y_center = y1 + 0.15 * (y2 - y1)

                # Pick target closest to crosshair (crop is 1:1 with screen pixels)
                dist = ((x_center - 320) ** 2 + (y_center - 320) ** 2) ** 0.5
                if dist < best_dist and dist > 3:   # dead zone: ignore <3px offsets
                    best_dist = dist
                    # coords from model are in 640x640 space, scale back to capture size
                    scale = CAPTURE_SIZE / 640
                    mx = (x_center - 320) * scale / 1.1
                    my = (y_center - 320) * scale / 1.1
                    # Smoothing: 0.75 = very snappy, reaches target in ~1 frame
                    smooth = 0.75
                    jx = random.uniform(-0.3, 0.3)
                    jy = random.uniform(-0.3, 0.3)
                    best_move   = (mx * smooth + jx, my * smooth + jy)
                    should_move = True

        with boxes_lock:
            latest_boxes = boxes_detected

        if aimbot_enabled and (right_click_pressed or left_click_pressed) and should_move:
            sleep(random.uniform(0.005, 0.015))   # 5-15 ms minimal reaction variance
            mouse_controller.move(round(best_move[0]), round(best_move[1]))

        # Dynamic rate: faster when aiming, slower when idle
        fps    = DETECT_FPS_ACTIVE if (right_click_pressed or left_click_pressed) else DETECT_FPS_IDLE
        period = 1.0 / fps
        elapsed = perf_counter() - t0
        wait = period - elapsed
        if wait > 0:
            sleep(wait)

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    global running
    try:
        detect_param = float(sys.argv[1])
    except Exception:
        detect_param = 0.60

    threading.Thread(target=mouse_listener_thread, daemon=True).start()
    threading.Thread(target=keyboard_listener_thread, daemon=True).start()
    threading.Thread(target=detection_loop, args=(detect_param,), daemon=True).start()
    print("Aimbot running — right/left-click to aim | F2 to toggle aimbot on/off")
    
    try:
        while running:
            sleep(1)
    except KeyboardInterrupt:
        running = False

main()
