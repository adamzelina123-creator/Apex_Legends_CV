from time import sleep
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
running             = True
esp_visible         = True   # Toggle with F1
latest_boxes        = []   # list of (x1, y1, x2, y2, conf)
boxes_lock          = threading.Lock()

# ── Mouse listener ─────────────────────────────────────────────────────────────
def on_mouse_press(x, y, button, pressed):
    global right_click_pressed
    if button == Button.right:
        right_click_pressed = pressed

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
def detection_loop(detect_param):
    global latest_boxes, running
    while running:
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
                y_center = y1 + 0.1 * (y2 - y1)

                # Pick target closest to crosshair (crop is 1:1 with screen pixels)
                dist = ((x_center - 320) ** 2 + (y_center - 320) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    # coords from model are in 640x640 space, scale back to capture size
                    scale = CAPTURE_SIZE / 640
                    mx = (x_center - 320) * scale / 1.5
                    my = (y_center - 320) * scale / 1.5
                    best_move   = (mx, my)
                    should_move = True

        with boxes_lock:
            latest_boxes = boxes_detected

        if right_click_pressed and should_move:
            mouse_controller.move(int(best_move[0]), int(best_move[1]))

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
            # Filled box with semi-transparent red tint (stipple) + bright outline
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2,
                                         fill='#FF0000', stipple='gray25',
                                         outline='#FF0000', width=4)
            # Bright white inner outline for contrast
            self.canvas.create_rectangle(sx1+4, sy1+4, sx2-4, sy2-4,
                                         outline='#FFFFFF', width=1)
            # Head circle
            hx = (sx1 + sx2) // 2
            self.canvas.create_oval(hx - 8, sy1 - 16, hx + 8, sy1,
                                    fill='#FF0000', outline='#FFFFFF', width=2)
            # Confidence label
            self.canvas.create_text(sx1, sy1 - 18,
                                    text=f'{conf:.0%}',
                                    fill='#FFFFFF',
                                    font=('Arial', 10, 'bold'),
                                    anchor='sw')

        self.root.after(16, self.update)   # ~60 fps

    def run(self):
        self.update()
        self.root.mainloop()

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    global running
    try:
        detect_param = float(sys.argv[1])
    except Exception:
        detect_param = 0.45

    threading.Thread(target=mouse_listener_thread, daemon=True).start()
    threading.Thread(target=keyboard_listener_thread, daemon=True).start()
    threading.Thread(target=detection_loop, args=(detect_param,), daemon=True).start()
    print("ESP ON  — press F1 to toggle overlay")

    esp = ESPOverlay()
    esp.run()

    running = False

main()