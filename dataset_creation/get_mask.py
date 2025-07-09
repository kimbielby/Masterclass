from Utils import get_filepaths, visuals
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os

def get_mask(top_dir, output_path_list, checkpoint, model_type, input_point=None):
    """
    Reads in each frame from a mov file (saved as png) and from that saves four things:
        1. The mask of the person
        2. The image that is inside the mask
        3. The image that is outside the mask.
        4. The Matplotlib figure
    Each will be saved in a different folder inside the output folder.
    :param model_type: SAM model type, probably 'vit_h'
    :param checkpoint: SAM model checkpoint
    :param top_dir: Where the frames as png files are located
    :param output_path_list: List of filepaths where each type of output will be saved
    """

    # Get full filepaths for each png frame in top_dir
    list_of_img_paths = get_full_filepaths(top_dir)

    # Load SAM model and create Predictor
    checkpoint = checkpoint
    model_type = model_type
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)

    for img_path in list_of_img_paths:
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create new filename for outputs
        base, _ = os.path.splitext(os.path.basename(img_path))
        mask_output_path = os.path.join(output_path_list[0], f"{base}_Mask.png")
        cutout_output_path = os.path.join(output_path_list[1], f"{base}_Cutout.png")
        background_output_path = os.path.join(output_path_list[2], f"{base}_Background.png")
        figure_output_path = os.path.join(output_path_list[3], f"{base}_Figure.png")

        # Get centre of image
        h, w = img.shape[:2]
        centre_x = w // 2
        centre_y = h // 2

        input_point = np.array([[centre_x + 100, centre_y + 500]])
        input_label = np.array([1])

        # Set current image for predictor
        predictor.set_image(img)

        # Get masks
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        # Display and save
        mask = masks[2]
        score = scores[2]
        # Plot
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        visuals.show_mask(mask, plt.gca())
        visuals.show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask 3, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        # Create rgba
        mask_uint8 = (mask.astype(np.uint8)) * 255
        inverse_mask_uint8 = cv2.bitwise_not(mask_uint8)
        rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        # Save Cutout
        person_cutout = rgba.copy()
        person_cutout[:, :, 3] = mask_uint8
        cv2.imwrite(cutout_output_path, person_cutout)
        # Save Background
        background_only = rgba.copy()
        background_only[:, :, 3] = inverse_mask_uint8
        cv2.imwrite(background_output_path, background_only)
        # Save Figure
        plt.savefig(figure_output_path, bbox_inches='tight', pad_inches=0)
        # Save Mask
        cv2.imwrite(mask_output_path, mask.astype(np.uint8) * 255)
        # Close plot
        plt.close()


def get_full_filepaths(top_dir):
    list_of_img_paths = get_filepaths(top_dir)
    for i in range(len(list_of_img_paths)):
        list_of_img_paths[i] = os.path.join(top_dir, list_of_img_paths[i])
    return list_of_img_paths








