import cv2
import os

def read_in_all_images(top_dir):
    """
    Reads in all images in all the subdirectories of a given directory
    :param top_dir: The directory where the subdirectories and images are located
    :return: List of images
    """
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
    """
    Reads in all images that live in one directory
    :param directory: Where the images live
    :return: List of images
    """
    list_of_images = []
    image_files = get_filepaths(directory)
    image_files.sort(key=str.lower)
    for img in image_files:
        img_path = os.path.join(directory, img)
        print(f"Reading in {img}")
        image = cv2.imread(img_path)  # keep as bgr
        # Resize
        image_rs = cv2.resize(image, (256, 256))
        list_of_images.append(image_rs)

    return list_of_images

def get_simple_image_filepaths(directory):
    """
    Gets the filepaths of all files that live in a given directory
    :param directory: Where the files live
    :return: A List of filepaths (Strings)
    """
    list_of_image_fpaths = []
    filepaths = get_filepaths(directory)
    filepaths.sort(key=str.lower)
    for img in filepaths:
        img_path = os.path.join(directory, img)
        list_of_image_fpaths.append(img_path)

    return list_of_image_fpaths

def get_simple_search_filepaths(directory, search_term):
    """
    Gets a list of filepaths within a given directory that contain the provided search term
    :param directory: Where the files live
    :param search_term: What should be part of the file name
    :return: List of filepaths (Strings)
    """
    list_of_image_fpaths = []
    filepaths = get_filepaths_search(dir_name=directory, search_term=search_term)
    filepaths.sort(key=str.lower)
    for img in filepaths:
        img_path = os.path.join(directory, img)
        list_of_image_fpaths.append(img_path)

    return list_of_image_fpaths

def get_filepaths(dir_name):
    """
    :param dir_name: Where the files live
    :return: A List of file names (Strings)
    """
    fpaths = [f for f in os.listdir(dir_name)]
    return fpaths

def get_filepaths_search(dir_name, search_term):
    """
    :param dir_name: Where the files live
    :param search_term: What should be part of the file name
    :return: A List of file names that contain the provided search term
    """
    fpaths = [f for f in os.listdir(dir_name) if search_term in f]
    return fpaths

