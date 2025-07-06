import cv2
import os

def read_in_all_images(top_dir):
    list_of_images = []

    # Get list of folder names under 'dset'
    filepaths = get_filepaths(dir_name=top_dir)

    # Replace each folder name with the root dir name and folder name
    for i in range(len(filepaths)):
        filepaths[i] = os.path.join(top_dir, filepaths[i])

    filepaths.sort(key=str.lower)

    # For each filepath to the sub-folder get filepaths to all images inside it
    for i in range(len(filepaths)):
        temp_fp = get_filepaths(dir_name=filepaths[i])
        print(f"Looking in directory {filepaths[i]}")
        if len(temp_fp) > 0:
            for j in range(len(temp_fp)):
                # Get filepath to image in subfolder
                temp_fp[j] = os.path.join(filepaths[i], temp_fp[j])
                # Load image - keep as bgr
                img = cv2.imread(temp_fp[j])
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Append to List
                list_of_images.append(img)

    return list_of_images

def read_in_images_simple(directory):
    list_of_images = []
    image_files = get_filepaths(directory)
    image_files.sort(key=str.lower)
    for img in image_files:
        img_path = os.path.join(directory, img)
        print(f"Reading in {img}")
        image = cv2.imread(img_path)  # keep as bgr
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize
        image_rs = cv2.resize(image, (256, 256))
        list_of_images.append(image_rs)

    return list_of_images

def get_filepaths(dir_name):
    fpaths = [f for f in os.listdir(dir_name)]
    return fpaths

