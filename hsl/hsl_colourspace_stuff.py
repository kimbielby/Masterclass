import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import cv2
import preprocessing.identify_face_etc as iden
import colorsys

def read_convert_hue(img_path):
    # Read in image
    image = cv2.imread(img_path)
    # Convert to HLS colour space
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # Extract Hue channel
    hls_image_hue = hls_image[:, :, 0]

    return hls_image, hls_image_hue

def normalise_hsl(h, s, l):
    return h / 360.0, s / 100.0, l / 100.0

def create_colour_heatmap(hsv_img, targ_hue=60, display=False, save_as=None):
    if save_as is None:
        save_as = "outputs/colour_heatmap.png"

    # Get HSV values
    hue = hsv_img[:, :, 0]
    sat = hsv_img[:, :, 1]
    val = hsv_img[:, :, 2]

    # Define target hue (60 is green)
    target_hue = targ_hue

    # Make array of same shape as hue filled with target hue
    target_hue_array = np.full_like(hue, target_hue)

    # Calculate distance image hue is from target hue
    hue_distance = cv2.absdiff(hue, target_hue_array)
    hue_distance = cv2.min(hue_distance, 180 - hue_distance)

    # Apply threshold skin_mask
    skin_mask = (sat > 50) & (val > 50)

    # Create spill mask (pixels far from skin hue, likely toward green
    spill_mask = (hue_distance > 10) & (hue_distance < 70) & skin_mask

    # Create image highlighting only  spill regions
    highlight = np.zeros_like(hue_distance, dtype=np.uint8)
    highlight[spill_mask] = hue_distance[spill_mask]

    # Normalise the hue distance to [0, 255] range
    highlight_norm = cv2.normalize(highlight, None, 0, 255, cv2.NORM_MINMAX)

    # Amplify small differences
    gama = 0.6
    highlight_float = (highlight_norm / 255.0) ** gama
    highlight_gama = (highlight_float * 255).astype(np.uint8)

    # Apply coloured heatmap: hues close to green are blue; hues further away are red
    coloured_heatmap = cv2.applyColorMap(highlight_gama, cv2.COLORMAP_JET)

    # Save the heatmap image
    cv2.imwrite(save_as, coloured_heatmap)

    # Display the heatmap image if display == True
    if display:
        plt.imshow(coloured_heatmap)

def create_bin_spill_mask(hsv_img, targ_hue=13, display=False, save_as=None):
    if save_as is None:
        save_as = "outputs/bin_spill_mask.png"

    # Get HSV values
    hue, sat, val = cv2.split(hsv_img)

    # Target hue for skin tone
    target_hue = targ_hue
    target_hue_array = np.full_like(hue, target_hue)

    # Absolute hue distance, circular
    diff = cv2.absdiff(hue, target_hue_array)
    hue_diff = cv2.min(diff, 180 - diff)

    # Mask for likely skin tones
    skin_mask = (sat > 50) & (val > 50)

    # Detect green spill regions
    spill_mask = (hue_diff > 30) & (hue_diff < 70) & skin_mask

    # Convert boolean mask to binary image (uint8)
    binary_mask = np.zeros_like(hue, dtype=np.uint8)
    binary_mask[spill_mask] = 255   # white for spill

    # Save mask
    cv2.imwrite(save_as, binary_mask)

    # Display mask
    if display:
        plt.imshow(binary_mask)

def replace_spill(hsv_img, targ_hue=13, display=False, save_as=None):
    if save_as is None:
        save_as = "outputs/despilled_image.png"

    # Create spill mask like above
    hue, sat, val = cv2.split(hsv_img)
    target_hue = targ_hue
    target_hue_array = np.full_like(hue, target_hue)
    diff = cv2.absdiff(hue, target_hue_array)
    hue_diff = cv2.min(diff, 180 - diff)
    skin_mask = (sat > 50) & (val > 50)
    spill_mask = (hue_diff > 30) & (hue_diff < 70) & skin_mask

    # Replace hue in spill areas
    corrected_hue = hue.copy()
    corrected_hue[spill_mask] = target_hue  # Assign new hue only where spill detected

    # Reduce saturation a bit to tone down green contamination
    # corrected_sat = sat.copy()
    # corrected_sat[spill_mask] = (corrected_sat[spill_mask] * 0.8).astype(np.uint8)

    # Merge corrected HSV and convert back to BGR
    corrected_hsv = cv2.merge([corrected_hue, sat, val])
    corrected_image = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)

    # Save despilled image
    cv2.imwrite(save_as, corrected_image)

    # Display despilled image
    if display:
        plt.imshow(corrected_image)

    return corrected_image

def visualise_hue_histogram(image_hue, save=False):
    # Calculate histogram of image hue
    histogram = cv2.calcHist([image_hue], [0], None, [180], [0, 180])

    # Get hue gradient scale
    hue_scale_rgb = generate_hue_gradient()

    # Plot histogram of hue
    plt.imshow(hue_scale_rgb, aspect='auto')
    plt.plot(range(180), 50 - histogram.flatten(), color='black', linewidth=2)
    plt.title('Histogram of Hue')
    plt.xlabel('Hue Value')
    plt.ylabel('Frequency')
    plt.xticks([0, 30, 60, 90, 120, 150, 180], ['Red', 'Yellow', 'Green', 'Cyan', 'Blue', 'Magenta', 'Red'])
    plt.yticks([])

    if save:
        plt.savefig('../outputs/hue_histogram.png')

    plt.show()

def remove_green_screen(og_img_path, save=False, display=False):
    # Load image
    og_img = cv2.imread(og_img_path)
    hls_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2HLS)

    # Define green hue range in HLS
    lower_green = np.array([35, 40, 40])  # H, L, S
    upper_green = np.array([85, 255, 255])

    # Create a mask where green pixels are white
    green_mask = cv2.inRange(hls_img, lower_green, upper_green)

    # Invert mask to keep everything that is NOT green
    mask_no_green = cv2.bitwise_not(green_mask)

    # Use inverted mask to extract non-green parts of og img
    result = cv2.bitwise_and(og_img, og_img, mask=mask_no_green)

    # Make green pixels fully transparent (RGBA output)
    result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result_rgba[green_mask == 255] = [0, 0, 0, 0]       # Set green areas to transparent

    if save:
        iden.save_img_as(fname='../outputs/img_sans_greenscreen.png', img=result_rgba)

    if display:
        plt.imshow(result_rgba)

    return result_rgba

### ABSTRACT HUE STUFF ###

def generate_hue_gradient():
    # Create 180 x 50 image to represent the different hues
    hue_scale = np.zeros((50, 180, 3), dtype=np.uint8)

    # Update each point on x-axis with a different hue
    for i in range(180):
        hue_scale[:, i] = [i, 255, 255]  # Hue, Max Saturation, Max Luminosity

    # Convert from HLS to RGB for display
    hue_scale_rgb = cv2.cvtColor(hue_scale, cv2.COLOR_HSV2RGB)

    return hue_scale_rgb

def visualise_hue_scale(save=False):
    # Get hue gradient
    hue_scale_rgb = generate_hue_gradient()

    # Display Hues
    plt.imshow(hue_scale_rgb)
    plt.xticks([0, 30, 60, 90, 120, 150, 180], ['Red', 'Yellow', 'Green', 'Cyan', 'Blue', 'Magenta', 'Red'])
    plt.yticks([])
    plt.title('Hue Bin to Colour Mapping')

    if save:
        plt.savefig('../outputs/hue_scale.png')

    plt.show()

def display_hue_swatches(hue_num):
    hue = hue_num
    hue_norm = hue / 360

    # Define saturation and luminosity levels
    sat_levels = [0.3, 0.5, 0.7]
    lum_levels = [0.3, 0.5, 0.7]

    # Build colour grid
    colours = []
    for lum in lum_levels:
        row = []
        for sat in sat_levels:
            rgb = colorsys.hls_to_rgb(hue_norm, lum, sat)
            row.append(rgb)
        colours.append(row)

    # Display
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(colours, extent=[0, 3, 0, 3])
    # xticks
    ax.set_xticks(np.arange(0.5, 3.5, 1))
    ax.set_xticklabels([f"S= {sat}" for sat in sat_levels])
    # yticks
    ax.set_yticks(np.arange(0.5, 3.5, 1))
    ax.set_yticklabels([f"L= {lum}" for lum in reversed(lum_levels)])
    # title
    ax.set_title(f"Hue {hue} in HSL - Variation of Saturation and Luminosity")
    plt.show()

def display_solid_hue(hue, sat, lum, save_as=None):
    hue, sat, lum = normalise_hsl(hue, sat, lum)

    # Turn hsl into rgb
    rgb = colorsys.hls_to_rgb(hue, lum, sat)

    fig, ax = plt.subplots(figsize=(3, 3))

    # Draw full sized square
    rect = patches.Rectangle((0, 0), 1, 1, facecolor=rgb)
    ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.box(False)

    # Save plot if filename provided
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0, dpi=300)

    # Display plot
    plt.show()







