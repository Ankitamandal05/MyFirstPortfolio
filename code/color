import cv2
import numpy as np

def detect_watch_color(image_path, box):
    image = cv2.imread(image_path)
    x1, y1, x2, y2 = map(int, box)
    cropped = image[y1:y2, x1:x2]

    # Convert to HSV color space
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Threshold for detecting black color
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))  # Low value, low saturation

    # Threshold for detecting silver color (grayish with high brightness and low saturation)
    silver_mask = cv2.inRange(hsv, (0, 0, 150), (180, 50, 255))  # Higher brightness, low saturation

    total_pixels = cropped.shape[0] * cropped.shape[1]
    
    # Calculate the ratio of pixels that are black or silver
    black_ratio = cv2.countNonZero(black_mask) / total_pixels
    silver_ratio = cv2.countNonZero(silver_mask) / total_pixels

    # Adjust the threshold for detection; if black or silver ratio is greater than 10% of the total pixels
    if black_ratio > silver_ratio and black_ratio > 0.1:
        return 'black'
    elif silver_ratio > black_ratio and silver_ratio > 0.1:
        return 'silver'
    else:
        return 'unknown'
