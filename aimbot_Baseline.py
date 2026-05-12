from time import time, sleep
# from utils.weapon_recoil_pattern import *
# from windowcapture1920x1080 import WindowCapture
# from windowcapture640x640 import WindowCapture
from pynput import keyboard
from pynput.mouse import Controller as MouseController, Listener as MouseListener, Button
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
from mss import mss
import numpy as np
import cv2 as cv
import sys
import threading

mouse_controller = MouseController()

print("//// LOADING MODEL ////")
model = YOLO('models/200923_best_yolov8n.pt')

# Use GPU if available for faster inference
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")
model.to(device)

# Open mss once globally instead of per-frame
_sct = mss()

def get_results():
    monitor = _sct.monitors[1]
    sct_img = _sct.grab(monitor)

    original_image = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    center_x = (original_image.width - 640) // 2
    center_y = (original_image.height - 640) // 2

    center_region = original_image.crop((center_x, center_y, center_x + 640, center_y + 640))

    results = model(center_region, verbose=False)
    return results

def update_global_variables(results):
    # global move_x, move_y
    try:
        # THIS ONLY TAKES THE FIRS [0] DETECTED CHARACTER - INCLUDE LOGIC TO TAKE IMAGE CLOSEST TO CROSSHAIR
        x_min = results[0].boxes.xyxy[0][0].item()
        y_min = results[0].boxes.xyxy[0][1].item()
        x_max = results[0].boxes.xyxy[0][2].item()
        y_max = results[0].boxes.xyxy[0][3].item()
        confidence = results[0].boxes.conf.item()
        cls = model.names[int(results[0].boxes.cls)]
        
        x_center = (x_min + x_max) / 2
        y_center = y_min + 0.1 * (y_max - y_min)
        
        # Calculate scaling factors
        scale_x = 1920 / 640
        scale_y = 1080 / 640

        # Translate to coordinates on the 1920x1080 image
        x_center_1920x1080 = x_center * scale_x
        y_center_1920x1080 = y_center * scale_y

        move_x = (x_center_1920x1080 - 960) / 1.61
        move_y = (y_center_1920x1080 - 540) / 1.61

        return move_x , move_y, confidence, cls
    
    except:
        print("NO DETECTION - skipping")
        move_x = 0
        move_y = 0
        confidence = 0
        cls = None

    return move_x, move_y, confidence, cls

# Global variable to track if right mouse button is held down
right_click_pressed = False

def on_mouse_press(x, y, button, pressed):
    global right_click_pressed
    if button == Button.right:
        right_click_pressed = pressed

def mouse_listener_thread():
    with MouseListener(on_click=on_mouse_press) as listener:
        listener.join()

def stop_aimbot():
    global running
    running = False


def mainloop():
    try:
        detect_param = float(sys.argv[1])
    except:
        detect_param = 0.45

    # Create threads for listeners
    caps_lock_thread = threading.Thread(target=mouse_listener_thread, daemon=True)

    # Start the listener threads
    caps_lock_thread.start()

    running = True

    previous_frame_time = 0
    fps_elapsed_time = 0
    
    while running:  # Run the main loop continuously
        if right_click_pressed:
            
            fps_elapsed_time = time() - previous_frame_time

            # Your code here, only executed when right mouse button is held down
            start_time = time()
                
            results = get_results()        
            move_x, move_y, confidence, cls = update_global_variables(results)
                
            if cls == 'avatar':
                if confidence is None:
                    pass
                elif confidence >= detect_param:
                    mouse_controller.move(int(move_x), int(move_y))
                    elapsed_time = time() - start_time
                    sleep(0.01)
        else:
            sleep(0.01)  # idle sleep — prevents CPU spin when not aiming


mainloop()
sys.exit()