from Utils import get_filepaths
from imports import *
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def get_auto_mask(top_dir, checkpoint, model_type):
    # Get full filepaths for each png frame in top_dir
    list_of_img_paths = get_full_filepaths(top_dir)

    # Load SAM model and create mask_generator
    sam_checkpoint = checkpoint
    model_type = model_type
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    #sam.to("cuda" if torch.cuda.is_available() else "cpu")
    mask_generator = SamAutomaticMaskGenerator(sam)

    for img_path in list_of_img_paths:
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create new filename for outputs
        base_folder = "../data/outputs"
        base, _ = os.path.splitext(os.path.basename(img_path))
        mask_output_path = f"masks/{base}_Mask_"
        # cutout_output_path = f"cutouts/{base}_Cutout_"

        # Get masks and sort by mask size
        masks = mask_generator.generate(img)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)

        # Check indices 2 and 3 exist
        if len(masks) > 3:

            # Read and save
            for i in [2, 3]:
                mask = masks[i]["segmentation"].astype(np.uint8) * 255

                # Get the right mask index folder
                folder = f"mask_{i}"

                # Get filenames sorted
                base_name = f"{mask_output_path}{i}.png"
                # cutout_name = f"{cutout_output_path}{i}.png"

                # Save mask
                cv2.imwrite((os.path.join(base_folder, folder, base_name)), mask)

                # Save cutout - Uncomment if needed
                # rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
                # rgba[:,:,3] = mask
                # cv2.imwrite(os.path.join(base_folder, folder, cutout_name), rgba)
        else:
            print(f"Not enough masks generated for {img_path}")


def get_full_filepaths(top_dir):
    list_of_img_paths = get_filepaths(top_dir)
    for i in range(len(list_of_img_paths)):
        list_of_img_paths[i] = os.path.join(top_dir, list_of_img_paths[i])
    return list_of_img_paths






