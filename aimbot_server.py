"""
aimbot_server.py — runs on your SECOND PC (any PC, no game needed)
Receives screen frames from the gaming PC, runs YOLO, sends back aim deltas.

Install on second PC:
    pip install ultralytics torch opencv-python numpy

Run:
    python aimbot_server.py
"""
import socket
import struct
import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# ── Model ──────────────────────────────────────────────────────────────────────
print("//// LOADING MODEL ////")
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

model  = YOLO(_model_path, task='detect')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")
if _model_path.endswith('.pt'):
    model.to(device)
torch.set_grad_enabled(False)
if device == 'cuda':
    torch.backends.cudnn.benchmark = True

_use_half = device == 'cuda' and _model_path.endswith('.pt')

HEAD_OFFSET    = 0.08
DETECT_CONF    = 0.55
TRACK_MAX_DIST = 120

# Warm up
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model(dummy, verbose=False, conf=DETECT_CONF, device=device, half=_use_half,
      max_det=10, agnostic_nms=True)
print("Model warmed up — ready")

HOST = '0.0.0.0'
PORT = 5005

# ── Helpers ────────────────────────────────────────────────────────────────────
def recv_all(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

# ── Client handler ─────────────────────────────────────────────────────────────
def handle_client(conn):
    tracked_cx = None
    tracked_cy = None

    while True:
        # ── Receive frame (4-byte length prefix + JPEG bytes) ──────────────────
        hdr = recv_all(conn, 4)
        if hdr is None:
            break
        length = struct.unpack('>I', hdr)[0]
        data   = recv_all(conn, length)
        if data is None:
            break

        # ── Decode and run YOLO ────────────────────────────────────────────────
        frame     = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = model(frame_rgb, verbose=False, conf=DETECT_CONF,
                          device=device, half=_use_half,
                          max_det=10, agnostic_nms=True)

        # ── Collect candidates ─────────────────────────────────────────────────
        candidates = []
        if len(results[0].boxes):
            for i, box in enumerate(results[0].boxes.xyxy):
                if model.names is not None:
                    if model.names[int(results[0].boxes.cls[i])] != 'avatar':
                        continue
                x1, y1, x2, y2 = (box[j].item() for j in range(4))
                candidates.append(((x1 + x2) / 2, y1 + HEAD_OFFSET * (y2 - y1)))

        # ── Tracking ───────────────────────────────────────────────────────────
        chosen_tx, chosen_ty = None, None
        if candidates:
            if tracked_cx is not None:
                best = min(candidates,
                           key=lambda c: (c[0] - tracked_cx) ** 2 + (c[1] - tracked_cy) ** 2)
                if ((best[0] - tracked_cx) ** 2 + (best[1] - tracked_cy) ** 2) ** 0.5 <= TRACK_MAX_DIST:
                    chosen_tx, chosen_ty = best
                else:
                    chosen_tx, chosen_ty = min(candidates,
                                               key=lambda c: (c[0] - 320) ** 2 + (c[1] - 320) ** 2)
            else:
                chosen_tx, chosen_ty = min(candidates,
                                           key=lambda c: (c[0] - 320) ** 2 + (c[1] - 320) ** 2)

        tracked_cx, tracked_cy = chosen_tx, chosen_ty

        dx = (chosen_tx - 320) if chosen_tx is not None else 0.0
        dy = (chosen_ty - 320) if chosen_ty is not None else 0.0

        # ── Send back delta ────────────────────────────────────────────────────
        conn.sendall(struct.pack('>ff', dx, dy))

# ── Main ───────────────────────────────────────────────────────────────────────
print(f"Server listening on port {PORT} — waiting for gaming PC...")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Gaming PC connected from {addr}")
        try:
            handle_client(conn)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()
        print("Disconnected — waiting for next connection...")
