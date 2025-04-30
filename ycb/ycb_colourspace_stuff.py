import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_ycb_img(img_ycb):
    y, cr, cb = cv2.split(img_ycb)
    cb = cb.astype(np.float32)
    cr = cr.astype(np.float32)
    return y, cr, cb

def create_green_spill_heatmap(img_ycb):
    y, cr, cb = split_ycb_img(img_ycb)

    # Calc green spill - high Cb, low Cr
    green_score = (cb - 130) - (cr - 120)
    green_score = np.clip(green_score, 0, None)

    # Normalise
    heatmap = cv2.normalize(green_score, None, 0, 1.0, cv2.NORM_MINMAX)

    # Boost low intensity green
    gamma = 0.4
    heatmap_gamma = np.power(heatmap, gamma)

    # Display heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_gamma, cmap='hot')
    plt.title('Green Spill Heatmap (Gamma Enhanced)')
    plt.axis('off')
    plt.colorbar(label='Spill Intensity')
    plt.show()

def quantify_green(img_ycb):
    y, cr, cb = split_ycb_img(img_ycb)
    # Define ranges
    lower_cb = 85
    upper_cb = 135
    lower_cr = 35
    upper_cr = 85

    # Create green pixel mask
    green_mask = (cb >= lower_cb) & (cb <= upper_cb) & (cr >= lower_cr) & (cr <= upper_cr)

    # Calculate percentage of green pixels
    total_pixels = img_ycb.shape[0] * img_ycb.shape[1]
    green_pixels = np.sum(green_mask)
    green_percentage = (green_pixels / total_pixels) * 100


    return green_pixels, green_percentage

