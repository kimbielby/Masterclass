import matplotlib.pyplot as plt
import numpy as np
import cv2
import Utils.identify_face_etc as iden

def read_convert_hue(img_path):
    # Read in image
    image = cv2.imread(img_path)
    # Convert to HLS colour space
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # Extract Hue channel
    hls_image_hue = hls_image[:, :, 0]

    return hls_image, hls_image_hue

# TODO
def create_colour_heatmap(image_hue, targ_hue=60, display=False, save_as=None):
    if save_as is None:
        save_as = "outputs/colour_heatmap.png"

    # Define target hue (60 is green)
    target_hue = targ_hue

    # Calculate distance image hue is from target hue
    hue_distance = cv2.absdiff(image_hue, target_hue)
    hue_distance = np.minimum(hue_distance, 180 - hue_distance)

    # Apply threshold mask
    mask = (hue_distance > 5) & (hue_distance < 15)

    # Create empty image and copy only values within threshold
    highlight = np.zeros_like(hue_distance, dtype=np.uint8)
    highlight[mask] = hue_distance[mask]

    # Normalise the hue distance to [0, 255] range
    normalised_hue_distance = cv2.normalize(highlight, None, 0, 255, cv2.NORM_MINMAX)

    # Amplify small differences
    gama = 0.5
    hue_distance_gama = np.power(normalised_hue_distance / 255.0, gama) * 255
    hue_distance_gama = hue_distance_gama.astype(np.uint8)

    # Apply coloured heatmap: hues close to green are blue; hues further away are red
    coloured_heatmap = cv2.applyColorMap(hue_distance_gama, cv2.COLORMAP_JET)

    # Save the heatmap image
    cv2.imwrite(save_as, coloured_heatmap)

    # Display the heatmap image if display == True
    if display:
        plt.imshow(coloured_heatmap)

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






