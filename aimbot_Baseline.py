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
import tkinter as tk
import torch

# ── Model ──────────────────────────────────────────────────────────────────────
print("//// LOADING MODEL ////")
model = YOLO('models/apex_trained.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")
model.to(device)

# ── Screen info ────────────────────────────────────────────────────────────────
_sct        = mss()
SCREEN_W    = 1600
SCREEN_H    = 900
# Capture a square region centred on the crosshair, scaled to 640 for the model.
# 640 = tight center, 900 = full screen height (max range for 900p)
CAPTURE_SIZE = min(900, SCREEN_W, SCREEN_H)   # 900 for 1600x900
CROP_X      = (SCREEN_W - CAPTURE_SIZE) // 2   # 350 — centres horizontally
CROP_Y      = (SCREEN_H - CAPTURE_SIZE) // 2   # 0   — full vertical coverage

# ── Aimbot tuning ──────────────────────────────────────────────────────────────
# AIM_FOV: max distance from crosshair (640px space) to lock onto a target.
#   Lower = only snap to enemies very close to crosshair (safer/more precise)
AIM_FOV         = 250
# AIM_STRENGTH: fraction of distance moved per frame. 0.3 = smooth tracking,
#   1.0 = instant snap (full distance every frame — hardest possible lock).
AIM_STRENGTH    = 1.0
# AIM_SENSITIVITY: divide movement by this to match your DPI × in-game sens.
#   Formula: in-game sens (5.0) × (DPI / 800) = 5.0 × (1200 / 800) = 7.5
#   Increase if overshooting, decrease if undershooting.
AIM_SENSITIVITY = 7.5
# AIM_HEAD_BIAS: fraction of box height from the top edge to the aim point.
#   0.0 = very top of box, 0.08 = head centre (head ≈ top 15% of full-body box)
AIM_HEAD_BIAS   = 0.08
# HEAD_SNAP_RADIUS: when the crosshair is this many screen-px from the head
#   centre, switch to instant (strength=1.0) to lock precisely on the head.
#   With AIM_STRENGTH=1.0 this is always hit — kept for when strength is lowered.
HEAD_SNAP_RADIUS = 50
# TARGET_LOCK_FRAMES: how many frames to keep locking the same target before
#   allowing a switch. Prevents flickering between enemies.
TARGET_LOCK_FRAMES = 12
# VELOCITY_WEIGHT: how much to lead the target based on its movement velocity.
#   0.0 = no prediction, 1.0 = full one-frame lead. 0.5 is a good start.
VELOCITY_WEIGHT = 0.5

# ── Detection quality filters ──────────────────────────────────────────────────
# CONF_ACQUIRE: confidence required to lock onto a brand-new target.
#   Higher = fewer false positives when acquiring. Raise if non-legend boxes fire.
CONF_ACQUIRE    = 0.55   # lowered — catch more targets, especially fast-moving ones
# CONF_MAINTAIN: relaxed threshold to keep tracking an already-confirmed target.
#   Prevents flickering on a target that briefly dips in confidence.
CONF_MAINTAIN   = 0.40
# Legend bounding-box aspect ratio (height / width) in 640-px model space.
#   Standing legends are taller than wide; filter everything else out.
ASPECT_MIN      = 1.2    # exclude wide blobs  (crates, vehicles, UI widgets)
ASPECT_MAX      = 4.5    # exclude tall slivers (poles, door frames, banners)
# Bounding-box height limits (640-px model space).
BOX_MIN_H       = 18     # ignore tiny distant blobs — likely noise or loot
BOX_MAX_H       = 560    # ignore near-full-screen boxes — usually walls/floor
# Temporal confirmation: a candidate must appear in the same ~20px grid cell
# for this many consecutive frames before it becomes a valid aim target.
#   2 = one frame of confirmation (very responsive yet filters single-frame hits)
CONFIRM_FRAMES  = 1   # 1 = no delay; target is valid the first frame it passes filters

# ── Shared state ───────────────────────────────────────────────────────────────
mouse_controller    = MouseController()
right_click_pressed = False
left_click_pressed  = False
running             = True
esp_visible         = True        # Toggle with F1
latest_boxes        = []          # list of (x1, y1, x2, y2, conf)
latest_aim_target   = None        # (screen_x, screen_y) of current aim point
boxes_lock          = threading.Lock()

# Target lock state (only used inside detection_loop)
_locked_target_id   = None        # (xc, yc) of locked target last frame
_lock_frames_left   = 0           # frames remaining on current lock
_target_vel         = (0.0, 0.0)  # velocity of locked target (px/frame)
_confirmation_buffer = {}         # (grid_x, grid_y) → consecutive frames seen
# Snap-on-click: track previous button state to detect the exact press frame
_prev_right_click   = False
_prev_left_click    = False

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

# ── Keyboard listener (F1 = toggle ESP) ───────────────────────────────────────
def on_key_press(key):
    global esp_visible
    if key == Key.f1:
        esp_visible = not esp_visible

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
        rgb = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
    return rgb

# ── Detection + aimbot loop ────────────────────────────────────────────────────
# Run as fast as the GPU allows — no artificial FPS cap.
# The game loop is the bottleneck; saturating detection only helps accuracy.
DETECT_FPS    = 60
DETECT_PERIOD = 1.0 / DETECT_FPS

def detection_loop(detect_param):
    global latest_boxes, latest_aim_target, running
    global _locked_target_id, _lock_frames_left, _target_vel, _confirmation_buffer
    global _prev_right_click, _prev_left_click
    while running:
        t0 = perf_counter()
        frame = get_frame()
        results = model(frame, verbose=False, conf=detect_param, device=device, half=(device == 'cuda'))

        # ── Pass 1: shape / size / confidence filters ──────────────────────────
        raw_detections = []
        if len(results[0].boxes):
            for i, box in enumerate(results[0].boxes.xyxy):
                conf     = results[0].boxes.conf[i].item()
                cls_name = model.names[int(results[0].boxes.cls[i])]
                if cls_name != 'avatar':
                    continue

                x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()

                # ── Shape / size guard ───────────────────────────────────────
                w = x2 - x1
                h = y2 - y1
                aspect = h / max(w, 1.0)
                if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
                    continue          # not a legend silhouette
                if not (BOX_MIN_H <= h <= BOX_MAX_H):
                    continue          # too small (noise) or too large (wall)

                xc   = (x1 + x2) / 2
                yc   = y1 + AIM_HEAD_BIAS * (y2 - y1)
                dist = ((xc - 320) ** 2 + (yc - 320) ** 2) ** 0.5
                if dist >= AIM_FOV:
                    continue

                # ── Confidence hysteresis ────────────────────────────────────
                # Allow a lower threshold to *keep* an already-locked target vs
                # a higher threshold to *acquire* a new one.
                is_near_lock = (
                    _lock_frames_left > 0 and _locked_target_id is not None
                    and ((xc - _locked_target_id[0]) ** 2
                         + (yc - _locked_target_id[1]) ** 2) ** 0.5 < 80
                )
                min_conf = CONF_MAINTAIN if is_near_lock else CONF_ACQUIRE
                if conf < min_conf:
                    continue

                raw_detections.append((xc, yc, x1, y1, x2, y2, conf, dist))

        # ── Pass 2: temporal confirmation ──────────────────────────────────────
        # A detection must appear in the same ~20 px grid cell for CONFIRM_FRAMES
        # consecutive frames before it is treated as a real target.
        new_buf      = {}
        avatar_boxes = []
        for det in raw_detections:
            xc, yc   = det[0], det[1]
            grid_key = (int(xc) // 20, int(yc) // 20)
            frames_seen = _confirmation_buffer.get(grid_key, 0) + 1
            new_buf[grid_key] = frames_seen
            if frames_seen >= CONFIRM_FRAMES:
                avatar_boxes.append(det)
        _confirmation_buffer = new_buf

        boxes_detected = [(a[2], a[3], a[4], a[5], a[6]) for a in avatar_boxes]

        # ── Target lock: persist on same enemy, update velocity each frame ─────
        best_move          = (0, 0)
        should_move        = False
        best_target_screen = None

        if avatar_boxes:
            target = None

            if _lock_frames_left > 0 and _locked_target_id is not None:
                lx, ly = _locked_target_id
                best_match = min(avatar_boxes,
                                 key=lambda a: (a[0]-lx)**2 + (a[1]-ly)**2)
                match_dist = ((best_match[0]-lx)**2 + (best_match[1]-ly)**2) ** 0.5
                if match_dist < 80:
                    # Update velocity: how much the target moved since last frame
                    _target_vel = (
                        0.6 * _target_vel[0] + 0.4 * (best_match[0] - lx),
                        0.6 * _target_vel[1] + 0.4 * (best_match[1] - ly),
                    )
                    _locked_target_id = (best_match[0], best_match[1])
                    _lock_frames_left -= 1
                    target = best_match
                else:
                    _lock_frames_left = 0

            if target is None:
                target = min(avatar_boxes, key=lambda a: a[7])
                _locked_target_id = (target[0], target[1])
                _lock_frames_left = TARGET_LOCK_FRAMES
                _target_vel = (0.0, 0.0)

            # Lead the target by predicted velocity
            pred_x = xc + _target_vel[0] * VELOCITY_WEIGHT
            pred_y = yc + _target_vel[1] * VELOCITY_WEIGHT

            scale = CAPTURE_SIZE / 640
            raw_x = (pred_x - 320) * scale / AIM_SENSITIVITY
            raw_y = (pred_y - 320) * scale / AIM_SENSITIVITY
            # Within HEAD_SNAP_RADIUS px: snap instantly so the crosshair
            # locks precisely onto the head instead of asymptotically approaching.
            dist_px = (raw_x ** 2 + raw_y ** 2) ** 0.5
            strength = 1.0 if dist_px <= HEAD_SNAP_RADIUS else AIM_STRENGTH
            best_move   = (raw_x * strength, raw_y * strength)
            should_move = abs(raw_x) > 0.5 or abs(raw_y) > 0.5
            best_target_screen = (
                int(pred_x * scale) + CROP_X,
                int(pred_y * scale) + CROP_Y,
            )
        else:
            _lock_frames_left = 0
            _locked_target_id = None
            _target_vel       = (0.0, 0.0)

        with boxes_lock:
            latest_boxes      = boxes_detected
            latest_aim_target = best_target_screen

        # Activate on right click (ADS) OR left click (shoot) — but only moves
        # the mouse when a confirmed legend is in the FOV (should_move guard).
        just_clicked = (
            (right_click_pressed and not _prev_right_click) or
            (left_click_pressed  and not _prev_left_click)
        )
        _prev_right_click = right_click_pressed
        _prev_left_click  = left_click_pressed

        if (right_click_pressed or left_click_pressed) and should_move:
            if just_clicked and best_target_screen is not None:
                # First frame of click: warp crosshair directly to head
                scale  = CAPTURE_SIZE / 640
                snap_x = round((target[0] - 320) * scale / AIM_SENSITIVITY)
                snap_y = round((target[1] - 320) * scale / AIM_SENSITIVITY)
                mouse_controller.move(snap_x, snap_y)
            else:
                dx, dy = round(best_move[0]), round(best_move[1])
                mouse_controller.move(dx, dy)

        # Cap detection rate — keeps GPU headroom for the game
        elapsed = perf_counter() - t0
        wait = DETECT_PERIOD - elapsed
        if wait > 0:
            sleep(wait)

# ── ESP Overlay ────────────────────────────────────────────────────────────────
class ESPOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('ESP')
        self.root.geometry(f'{SCREEN_W}x{SCREEN_H}+0+0')
        self.root.overrideredirect(True)
        self.root.wm_attributes('-topmost', True)
        self.root.wm_attributes('-transparentcolor', 'black')
        self.root.configure(bg='black')
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0,
                                width=SCREEN_W, height=SCREEN_H)
        self.canvas.pack()

    def update(self):
        self.canvas.delete('all')
        if not esp_visible:
            self.root.after(16, self.update)
            return
        with boxes_lock:
            boxes = list(latest_boxes)

        for (x1, y1, x2, y2, conf) in boxes:
            scale = CAPTURE_SIZE / 640
            sx1 = int(x1 * scale) + CROP_X
            sy1 = int(y1 * scale) + CROP_Y
            sx2 = int(x2 * scale) + CROP_X
            sy2 = int(y2 * scale) + CROP_Y

            cx     = (sx1 + sx2) // 2
            head_y = sy1 + int((sy2 - sy1) * AIM_HEAD_BIAS)

            # ── Body line ────────────────────────────────────────────────────
            self.canvas.create_line(cx, sy2, cx, head_y,
                                    fill='#FF2200', width=1, dash=(4, 4))

            # ── Head glow: 2 rings only to reduce GPU load ────────────────
            self.canvas.create_oval(cx - 16, head_y - 16,
                                    cx + 16, head_y + 16,
                                    outline='#880000', width=3, fill='')
            self.canvas.create_oval(cx - 7, head_y - 7,
                                    cx + 7, head_y + 7,
                                    outline='#FF4400', width=2, fill='')
            self.canvas.create_oval(cx - 2, head_y - 2,
                                    cx + 2, head_y + 2,
                                    fill='#FFFFFF', outline='')

        # ── Aim crosshair: only visible while right-clicking (ADS) ───────────
        if right_click_pressed:
            with boxes_lock:
                aim = latest_aim_target
            if aim:
                tx, ty = aim
                self.canvas.create_line(tx - 10, ty, tx + 10, ty,
                                        fill='#00FF88', width=1)
                self.canvas.create_line(tx, ty - 10, tx, ty + 10,
                                        fill='#00FF88', width=1)
                self.canvas.create_oval(tx - 4, ty - 4, tx + 4, ty + 4,
                                        outline='#00FF88', width=1, fill='')

        self.root.after(50, self.update)   # 20 fps — reduces game lag

    def run(self):
        self.update()
        self.root.mainloop()

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
    print("ESP ON  — press F1 to toggle overlay")

    esp = ESPOverlay()
    esp.run()

    running = False

main()
