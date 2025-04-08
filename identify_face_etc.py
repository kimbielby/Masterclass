import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Set Colours
BG_COLOUR = (192, 192, 192)
MASK_COLOUR = (255, 255, 255)

# Define category labels
labels = {
    "Background": 0,
    "Hair": 1,
    "Body Skin": 2,
    "Face Skin": 3,
    "Clothes": 4,
    "Others (Accessories)": 5
}

def here_we_go(og_img, class_indices=None):
    if class_indices is None:
        class_indices = [2, 3]

    # Create options for segmenter
    model_path = "models/selfie_multiclass_256x256.tflite"
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                           output_category_mask=True,
                                           output_confidence_masks=False)

    # Load og image
    og_image = cv2.imread(og_img)
    og_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)

    # Create Image Segmenter
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        # Load input image
        input_image = mp.Image.create_from_file(og_img)
    
        # Segment
        segmentation_result = segmenter.segment(input_image)

        # Get category mask
        category_mask = np.array(segmentation_result.category_mask.numpy_view(), dtype=np.uint8)

        # Create binary mask for selected class indices
        combined_mask = np.isin(category_mask, class_indices).astype(np.uint8) * 255

        # Apply mask to og image
        segmented_combined = cv2.bitwise_and(og_image, og_image, mask=combined_mask)

    return combined_mask, segmented_combined


def save_as(fname, img):
    cv2.imwrite(filename=fname, img=img)

