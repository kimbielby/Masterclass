import cv2
import os
import numpy as np

def is_greyscale_hsv(img_path, saturation_threshold=10):
    image = cv2.imread(img_path)
    if image is None:
        return False

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]    # Saturation channel

    mean_saturation = np.mean(s)
    is_grey = mean_saturation < saturation_threshold

    print(f"{img_path} - Mean Saturation: {mean_saturation} -> Greyscale: {is_grey}")

    return is_grey

def is_sepia_lab(img_path, b_threshold=20, a_threshold=10):
    image = cv2.imread(img_path)
    if image is None:
        return False

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    mean_a = np.mean(a)
    mean_b = np.mean(b)

    is_sepia = mean_b > (128 + b_threshold) and mean_a > (128 + a_threshold)

    print(f"{img_path} - Mean a: {mean_a}, Mean b: {mean_b} -> Sepia: {is_sepia}")

    return is_sepia

results = []
def is_mostly_greyscale(img_path, pixel_ratio_threshold=0.95, pixel_saturation_threshold=20):
    image = cv2.imread(img_path)
    if image is None:
        return False

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    low_sat_pixels = np.sum(s < pixel_saturation_threshold)
    total_pixels = s.size
    ratio = low_sat_pixels / total_pixels

    is_mostly_grey = ratio > pixel_ratio_threshold

    result = {
        "filename": os.path.basename(img_path),
        "mean_saturation": np.mean(s),
        "low_sat_ratio": ratio,
        "is_mostly_grey": is_mostly_grey
    }
    results.append(result)

    print(f"{img_path} - {ratio*100:.1f}% low-saturation pixels -> Mostly Greyscale: {is_mostly_grey}")
    return is_mostly_grey

def get_results_list():
    return results


