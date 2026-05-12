from time import sleep
from pynput.mouse import Controller as MouseController, Listener as MouseListener, Button
from ultralytics import YOLO
from mss import mss
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
_sct     = mss()
_monitor = _sct.monitors[1]
SCREEN_W = _monitor['width']
SCREEN_H = _monitor['height']
CROP_X   = (SCREEN_W - 640) // 2
CROP_Y   = (SCREEN_H - 640) // 2

# ── Shared state ───────────────────────────────────────────────────────────────
mouse_controller    = MouseController()
right_click_pressed = False
running             = True
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

# ── Fast screen capture (numpy, no PIL) ────────────────────────────────────────
def get_frame():
    sct_img = _sct.grab(_monitor)
    img = np.frombuffer(sct_img.bgra, dtype=np.uint8).reshape(SCREEN_H, SCREEN_W, 4)
    # Crop center 640x640 and convert BGRA→RGB
    crop = img[CROP_Y:CROP_Y+640, CROP_X:CROP_X+640, :3][..., ::-1].copy()
    return crop

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
                    mx = (x_center - 320) / 1.5
                    my = (y_center - 320) / 1.5
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
        with boxes_lock:
            boxes = list(latest_boxes)

        for (x1, y1, x2, y2, conf) in boxes:
            sx1 = int(x1) + CROP_X
            sy1 = int(y1) + CROP_Y
            sx2 = int(x2) + CROP_X
            sy2 = int(y2) + CROP_Y
            # Bounding box
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2,
                                         outline='#FF0000', width=2)
            # Head circle
            hx = (sx1 + sx2) // 2
            self.canvas.create_oval(hx - 6, sy1 - 12, hx + 6, sy1,
                                    outline='#FF0000', width=2)
            # Confidence label
            self.canvas.create_text(sx1, sy1 - 14,
                                    text=f'{conf:.0%}',
                                    fill='#FF0000',
                                    font=('Arial', 9, 'bold'),
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
    threading.Thread(target=detection_loop, args=(detect_param,), daemon=True).start()

    esp = ESPOverlay()
    esp.run()

    running = False

main()