from time import sleep, perf_counter
import ctypes
import random
from pynput.mouse import Listener as MouseListener, Button
from pynput.keyboard import Listener as KeyboardListener, Key
from ultralytics import YOLO
import dxcam
import numpy as np
import sys
import threading
import torch

# ── Process priority (below-normal so Apex always gets CPU first) ─────────────
ctypes.windll.kernel32.SetPriorityClass(
    ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS
)

# ── Model ──────────────────────────────────────────────────────────────────────
print("//// LOADING MODEL ////")
import os
_engine = 'models/200923_best_yolov8n.engine'
_onnx   = 'models/200923_best_yolov8n.onnx'
_pt     = 'models/200923_best_yolov8n.pt'
if os.path.exists(_engine):
    _model_path = _engine
elif os.path.exists(_onnx):
    _model_path = _onnx
else:
    _model_path = _pt
print(f"Using: {_model_path}")
model = YOLO(_model_path, task='detect')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")
if _model_path.endswith('.pt'):
    model.to(device)
torch.set_num_threads(1)
torch.set_grad_enabled(False)
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.30)
# half precision: .pt on CUDA only; .onnx exported as FP32; .engine has it baked in
_use_half = device == 'cuda' and _model_path.endswith('.pt')

# ── Screen info ────────────────────────────────────────────────────────────────
SCREEN_W = ctypes.windll.user32.GetSystemMetrics(0)
SCREEN_H = ctypes.windll.user32.GetSystemMetrics(1)
# 640 = direct capture at model resolution — zero resize cost
CAPTURE_SIZE = 640
CROP_X       = (SCREEN_W  - CAPTURE_SIZE) // 2
CROP_Y       = (SCREEN_H  - CAPTURE_SIZE) // 2
CENTER       = CAPTURE_SIZE // 2   # 320
_camera = dxcam.create(output_color='RGB')
_region = (CROP_X, CROP_Y, CROP_X + CAPTURE_SIZE, CROP_Y + CAPTURE_SIZE)

# ── Aim tuning ─────────────────────────────────────────────────────────────────
# HEAD_OFFSET: fraction from bbox top → head point (0.0 = very top, 0.15 = mid-upper)
HEAD_OFFSET = 0.08
# SNAP_RATIO: fraction of remaining distance covered per aim tick.
# 0.80 at 120 Hz = ~2 ticks (~16 ms) to cover 96% of distance — near-instant snap.
SNAP_RATIO  = 0.80
AIM_HZ      = 120   # dedicated aim loop frequency

# ── Raw mouse move (works with Apex raw input) ────────────────────────────────
def raw_move(dx, dy):
    ctypes.windll.user32.mouse_event(0x0001, int(dx), int(dy), 0, 0)

# ── Shared state ───────────────────────────────────────────────────────────────
right_click_pressed = False
left_click_pressed  = False
running             = True
aimbot_enabled      = True   # Toggle with F2

# Remaining delta (screen pixels) to the head target.
# detection_loop sets these; aim_loop consumes them incrementally.
_aim_dx   = 0.0
_aim_dy   = 0.0
_aim_lock = threading.Lock()

# Tracking state — last known head position of the locked target (model coords)
# If None, will acquire closest-to-center on next detection frame.
_tracked_cx  = None
_tracked_cy  = None
_track_lock  = threading.Lock()
TRACK_MAX_DIST = 120   # pixels; if target moves further it's considered lost

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

# ── Keyboard listener (F2 = toggle aimbot) ────────────────────────────────────
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
    frame = _camera.grab(region=_region)   # returns RGB ndarray directly, no copy needed
    if frame is None:                       # dxcam returns None if frame hasn't changed
        return None
    return frame

# ── Detection loop ─────────────────────────────────────────────────────────────
# Runs YOLO as fast as the GPU allows.
# Tracking: on first detection acquires the closest-to-centre target; on
# subsequent frames keeps the same target by matching to its last head position.
def detection_loop(detect_param):
    global _aim_dx, _aim_dy, _tracked_cx, _tracked_cy, running

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, verbose=False, conf=detect_param, device=device, half=_use_half,
          max_det=10, agnostic_nms=True)
    print("Model warmed up — ready")

    while running:
        if not aimbot_enabled:
            with _aim_lock:
                _aim_dx = 0.0
                _aim_dy = 0.0
            with _track_lock:
                _tracked_cx = None
                _tracked_cy = None
            sleep(0.05)
            continue

        frame = get_frame()
        if frame is None:   # dxcam: no new frame yet, skip
            continue
        results = model(frame, verbose=False, conf=detect_param,
                        device=device, half=_use_half,
                        max_det=10, agnostic_nms=True)

        # Collect all avatar head candidates this frame
        candidates = []   # list of (tx, ty)
        if len(results[0].boxes):
            for i, box in enumerate(results[0].boxes.xyxy):
                # Skip class check when model.names is None (ONNX) — model only detects avatars
                if model.names is not None:
                    cls_name = model.names[int(results[0].boxes.cls[i])]
                    if cls_name != 'avatar':
                        continue
                x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                tx = (x1 + x2) / 2
                ty = y1 + HEAD_OFFSET * (y2 - y1)
                candidates.append((tx, ty))

        chosen_tx, chosen_ty = None, None

        if candidates:
            with _track_lock:
                tcx, tcy = _tracked_cx, _tracked_cy

            if tcx is not None:
                # Continue tracking: find candidate closest to last known position
                best = min(candidates, key=lambda c: (c[0]-tcx)**2 + (c[1]-tcy)**2)
                dist_to_track = ((best[0]-tcx)**2 + (best[1]-tcy)**2) ** 0.5
                if dist_to_track <= TRACK_MAX_DIST:
                    chosen_tx, chosen_ty = best
                else:
                    # Lost track — reacquire closest to centre
                    chosen_tx, chosen_ty = min(candidates,
                        key=lambda c: (c[0]-CENTER)**2 + (c[1]-CENTER)**2)
            else:
                # No active track — acquire closest to centre
                chosen_tx, chosen_ty = min(candidates,
                    key=lambda c: (c[0]-CENTER)**2 + (c[1]-CENTER)**2)

        with _track_lock:
            _tracked_cx = chosen_tx
            _tracked_cy = chosen_ty

        with _aim_lock:
            if chosen_tx is not None:
                _aim_dx = chosen_tx - CENTER
                _aim_dy = chosen_ty - CENTER
            else:
                _aim_dx = 0.0
                _aim_dy = 0.0

# ── Aim loop ───────────────────────────────────────────────────────────────────
# Runs at AIM_HZ independent of YOLO speed.  Each tick it covers SNAP_RATIO of
# the remaining distance, giving fast snap + continuous lock without overshoot.
def aim_loop():
    global _aim_dx, _aim_dy
    interval = 1.0 / AIM_HZ

    while running:
        t0 = perf_counter()

        if (right_click_pressed or left_click_pressed) and aimbot_enabled:
            with _aim_lock:
                mx = _aim_dx * SNAP_RATIO
                my = _aim_dy * SNAP_RATIO
                # Reduce remaining delta so we don't overshoot between detections
                _aim_dx -= mx
                _aim_dy -= my

            # Only fire if the move is at least half a pixel
            if abs(mx) >= 0.5 or abs(my) >= 0.5:
                raw_move(round(mx), round(my))

        elapsed = perf_counter() - t0
        wait = interval - elapsed
        if wait > 0:
            sleep(wait)

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    global running
    try:
        detect_param = float(sys.argv[1])
    except Exception:
        detect_param = 0.55

    threading.Thread(target=mouse_listener_thread,    daemon=True).start()
    threading.Thread(target=keyboard_listener_thread, daemon=True).start()
    threading.Thread(target=detection_loop, args=(detect_param,), daemon=True).start()
    threading.Thread(target=aim_loop,                 daemon=True).start()
    print("Aimbot running — right/left-click to aim | F2 to toggle")

    try:
        while running:
            sleep(1)
    except KeyboardInterrupt:
        running = False

main()
