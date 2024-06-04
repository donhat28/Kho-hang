import cv2
import numpy as np

class ColorDetector:
    def __init__(self):
        self.color_ranges = self.define_color_range()

    def define_color_range(self):
        orange_lower = np.array([10, 100, 20])
        orange_upper = np.array([25, 255, 255])

        yellow_lower = np.array([30, 100, 20])
        yellow_upper = np.array([60, 255, 255])

        green_lower = np.array([90, 100, 20])
        green_upper = np.array([120, 255, 255])

        return {
            'orange': (orange_lower, orange_upper),
            'yellow': (yellow_lower, yellow_upper),
            'green': (green_lower, green_upper)
        }

    def detect_color_in_mask(self, masked_hsv):
        for color, (lower, upper) in self.color_ranges.items():
            color_mask = cv2.inRange(masked_hsv, lower, upper)

            if np.any(color_mask > 0):
                return color
        return None