import numpy as np
from mss import mss as _mss

class WindowCapture:
    w = 1920
    h = 1080
    offset_x = 0
    offset_y = 0

    def __init__(self, window_name=None):
        # On Linux we capture the full primary monitor regardless of window_name
        with _mss() as sct:
            monitor = sct.monitors[1]
            self.w = monitor['width']
            self.h = monitor['height']
            self.offset_x = monitor['left']
            self.offset_y = monitor['top']

    def get_screenshot(self):
        with _mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            img = img[..., :3]  # drop alpha, keep BGR
            img = np.ascontiguousarray(img)
            return img

    @staticmethod
    def list_window_names():
        print("list_window_names is not supported on Linux.")

    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)