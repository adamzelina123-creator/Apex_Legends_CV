import cv2 as cv
from time import sleep, time
from utils.windowcapture1920x1080 import WindowCapture 
import os 

def detect_what_weapon():
    sleep(0.4)
    global detected_weapon
    wincap = WindowCapture('Apex Legends')
    screenshot = wincap.get_screenshot()

    weapons = [file for file in os.listdir('templates') if file.lower().endswith('.jpg')]

    best_name  = None
    best_score = 0.0

    for weapon in weapons:
        template = cv.imread(f'templates/{weapon}', cv.IMREAD_COLOR)
        if template is None:
            continue

        method = cv.TM_CCOEFF_NORMED
        img2   = screenshot.copy()
        result = cv.matchTemplate(img2, template, method)
        _, max_val, _, _ = cv.minMaxLoc(result)

        print(f"  {str(weapon).split('.jpg')[0]}: {max_val:.3f}")

        if max_val > best_score:
            best_score = max_val
            best_name  = str(weapon).split('.jpg')[0]

    if best_score > 0.85:
        detected_weapon = best_name
        print(f"DETECTED: {detected_weapon} (score {best_score:.3f})")
        return detected_weapon
    else:
        print(f"No weapon detected (best: {best_name} @ {best_score:.3f}) — scroll in-game to retry")
        return None
