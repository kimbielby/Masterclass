import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

"""
category labels = {
    "Background": 0,
    "Hair": 1,
    "Body Skin": 2,
    "Face Skin": 3,
    "Clothes": 4,
    "Others (Accessories)": 5
}
"""

def segment_face_et_al(og_img_path, class_indices=None):
    if class_indices is None:
        class_indices = [2, 3]

    # Create options for segmenter
    model_path = "../preprocessing/models/selfie_multiclass_256x256.tflite"
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                           output_category_mask=True,
                                           output_confidence_masks=False)

    # Get og image dimensions
    path = og_img_path
    print(f"File exists: {os.path.exists(path)}")
    og_image = cv2.imread(og_img_path)
    if og_image is None:
        print(f"Unable to load {og_img_path}")
    else:
        print(f"Loaded {og_img_path}")

    h, w = og_image.shape[:2]

    # Create Image Segmenter
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        # Load input image
        input_image = mp.Image.create_from_file(og_img_path)
        # Segment
        segmentation_result = segmenter.segment(input_image)
        # Get category mask
        category_mask = segmentation_result.category_mask.numpy_view()
        # Resize mask to og image size
        category_mask_resized = cv2.resize(src=category_mask,
                                           dsize=(w, h),
                                           interpolation=cv2.INTER_NEAREST)
        # Choose which classes to keep
        mask = np.isin(category_mask_resized, class_indices)

        # Apply mask to og image
        mask_3ch = np.stack([mask]*3, axis=-1)
        output_image = np.where(mask_3ch, og_image, 0)

    return output_image



