"""
aimbot_client.py — runs on YOUR GAMING PC
Captures the screen, sends frames to the server PC, receives aim deltas, moves mouse.
Zero GPU inference — the gaming GPU is 100% free for Apex.

Install on gaming PC:
    pip install dxcam opencv-python numpy pynput

Usage:
    1. Find your second PC's local IP (run `ipconfig` on it, look for IPv4)
    2. Set SERVER_IP below to that IP
    3. Start aimbot_server.py on the second PC FIRST
    4. Run this script: python aimbot_client.py

Controls:
    Right-click or left-click = aim
    F2 = toggle aimbot on/off
"""
from time import sleep, perf_counter
import ctypes
import socket
import struct
import cv2
import numpy as np
import threading
import dxcam
from pynput.mouse import Listener as MouseListener, Button
from pynput.keyboard import Listener as KeyboardListener, Key

# ── CONFIG — change SERVER_IP to your second PC's local IP ────────────────────
SERVER_IP   = '192.168.2.71'     # old PC's IP
SERVER_PORT = 5005
CAPTURE_SIZE   = 640
JPEG_QUALITY   = 90    # higher = better detection on server (GPU free now)
SNAP_RATIO     = 0.45  # aim smoothing: 0.45 smooth, 1.0 instant
CAPTURE_FPS_ACTIVE = 30  # max grabs/sec while aiming — limits dxcam VRAM pressure
CAPTURE_FPS_IDLE   = 10  # max grabs/sec while idle

# ── Process priority ───────────────────────────────────────────────────────────
ctypes.windll.kernel32.SetPriorityClass(
    ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000
)

# ── Screen capture ─────────────────────────────────────────────────────────────
SCREEN_W = ctypes.windll.user32.GetSystemMetrics(0)
SCREEN_H = ctypes.windll.user32.GetSystemMetrics(1)
CROP_X   = (SCREEN_W - CAPTURE_SIZE) // 2
CROP_Y   = (SCREEN_H - CAPTURE_SIZE) // 2
_camera  = dxcam.create(output_color='BGR')   # BGR so cv2 can JPEG-encode directly
_region  = (CROP_X, CROP_Y, CROP_X + CAPTURE_SIZE, CROP_Y + CAPTURE_SIZE)

# ── Raw mouse move (works with Apex raw input) ────────────────────────────────
def raw_move(dx, dy):
    ctypes.windll.user32.mouse_event(0x0001, int(dx), int(dy), 0, 0)

# ── Shared state ───────────────────────────────────────────────────────────────
right_click_pressed = False
left_click_pressed  = False
aimbot_enabled      = True
running             = True

def on_mouse_press(x, y, button, pressed):
    global right_click_pressed, left_click_pressed
    if button == Button.right:
        right_click_pressed = pressed
    elif button == Button.left:
        left_click_pressed = pressed

def on_key_press(key):
    global aimbot_enabled
    if key == Key.f2:
        aimbot_enabled = not aimbot_enabled
        print(f"Aimbot {'ENABLED' if aimbot_enabled else 'DISABLED'}")

threading.Thread(
    target=lambda: MouseListener(on_click=on_mouse_press).start(), daemon=True).start()
threading.Thread(
    target=lambda: KeyboardListener(on_press=on_key_press).start(), daemon=True).start()

# ── Network helper ─────────────────────────────────────────────────────────────
def recv_all(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Server disconnected")
        buf.extend(chunk)
    return bytes(buf)

# ── Main loop ──────────────────────────────────────────────────────────────────
# Remaining aim delta — consumed each iteration for smooth movement
_aim_dx = 0.0
_aim_dy = 0.0

print(f"Connecting to {SERVER_IP}:{SERVER_PORT} ...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, SERVER_PORT))
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # disable Nagle algorithm
print("Connected — aimbot running | right/left-click to aim | F2 to toggle")

try:
    while running:
        t0 = perf_counter()

        # ── Capture ────────────────────────────────────────────────────────────
        frame = _camera.grab(region=_region)
        if frame is None:
            sleep(0.001)
            continue

        # ── Encode and send to server ─────────────────────────────────────────
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        data    = jpeg.tobytes()
        sock.sendall(struct.pack('>I', len(data)) + data)

        # ── Receive aim delta from server ─────────────────────────────────────
        response   = recv_all(sock, 8)
        raw_dx, raw_dy = struct.unpack('>ff', response)

        # Update remaining delta (new detection resets it)
        _aim_dx = raw_dx
        _aim_dy = raw_dy

        # ── Move mouse if button held ─────────────────────────────────────────
        if (right_click_pressed or left_click_pressed) and aimbot_enabled:
            mx = _aim_dx * SNAP_RATIO
            my = _aim_dy * SNAP_RATIO
            if abs(mx) >= 0.5 or abs(my) >= 0.5:
                raw_move(round(mx), round(my))
        # ── Rate-limit capture to free GPU for Apex texture streaming ─────────
        fps  = CAPTURE_FPS_ACTIVE if (right_click_pressed or left_click_pressed) else CAPTURE_FPS_IDLE
        wait = 1.0 / fps - (perf_counter() - t0)
        if wait > 0:
            sleep(wait)
except KeyboardInterrupt:
    print("Stopped.")
except Exception as e:
    print(f"Error: {e}")
finally:
    sock.close()
