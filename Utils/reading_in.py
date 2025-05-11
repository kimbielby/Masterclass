import cv2
import os

def read_in_all_images(top_dir):
    list_of_images = []

    # Get list of folder names under 'dset'
    filepaths = get_filepaths(dir_name=top_dir)

    # Replace each folder name with the root dir name and folder name
    for i in range(len(filepaths)):
        filepaths[i] = os.path.join(top_dir, filepaths[i])

    # For each filepath to the sub-folder get filepaths to all images inside it
    for i in range(len(filepaths)):
        temp_fp = get_filepaths(dir_name=filepaths[i])
        for j in range(len(temp_fp)):
            # Get filepath to image in subfolder
            temp_fp[j] = os.path.join(filepaths[i], temp_fp[j])
            # Load image
            img = cv2.imread(temp_fp[j])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Append to List
            list_of_images.append(img)

    return list_of_images

def read_in_images_simple(folder):
    list_of_images = []
    image_files = get_filepaths(folder)
    for img in image_files:
        img_path = os.path.join(folder, img)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            list_of_images.append(img)

    return list_of_images

def get_filepaths(dir_name):
    fpaths = [f for f in os.listdir(dir_name)]
    return fpaths

