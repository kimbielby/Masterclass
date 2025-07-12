import cv2
import numpy as np

def add_channels(bgr_img):
    """
    Adds extra channels to the image (hue and greenness)
    :param bgr_img: BGR image
    :return: Image ready to be turned into a PyTorch tensor
    """
    bgr = bgr_img.astype(np.float32) / 255.0
    # Split BGR channels
    b, g, r = cv2.split(bgr)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = h.astype(np.float32) / 180.0    # Normalise to  [0, 1]

    # Compute "greenness" metric
    greenness = g - 0.5 * (r + b)
    greenness = np.clip(greenness, 0.0, 1.0)

    # Stack into a (H, W, 5) input tensor
    input_tensor = np.stack([r, g, b, h, greenness], axis=-1)

    return input_tensor



