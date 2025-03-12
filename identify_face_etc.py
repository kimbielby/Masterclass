import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Set Colours
BG_COLOUR = (192, 192, 192)
MASK_COLOUR = (255, 255, 255)

def here_we_go(og_img, label="Face Skin"):
    # Create options for segmenter
    model_path = "models/selfie_multiclass_256x256.tflite"
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                           output_category_mask=True,
                                           output_confidence_masks=False)

    # Create Image Segmenter
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        # Load input image
        input_image = mp.Image.create_from_file(og_img)
    
        # Segment
        segmentation_result = segmenter.segment(input_image)

        # Get category mask
        category_mask = np.array(segmentation_result.category_mask.numpy_view())

    # Define category labels
    labels = {
        "Background": 0,
        "Hair": 1,
        "Body Skin": 2,
        "Face Skin": 3,
        "Clothes": 4,
        "Others (Accessories)": 5
    }

    # Select a label
    selected_label = label
    selected_index = labels[selected_label]

    # Load og image
    og_image = cv2.imread(og_img)
    og_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)

    # Create binary mask
    binary_mask = (category_mask == selected_index).astype(np.uint8) * 255

    # Apply mask to og image
    segmented_image = cv2.bitwise_and(og_image, og_image, mask=binary_mask)

    return binary_mask, segmented_image


def save_as(fname, img):
    cv2.imwrite(filename=fname, img=img)

