import os
import cv2

def save_img_as(fname, img):
    cv2.imwrite(filename=fname, img=img)

def get_unique_filename(dest_dir, fname):
    """
    When moving an image to a new directory, it checks there is not already a file in
    their with the same name. If there is, it adds a number to the end of the filename
    :param dest_dir: Where the file is being moved to
    :param fname: The current name of the file
    :return: Updated filename or og filename, depending
    """
    base, ext = os.path.splitext(fname)
    counter = 1
    new_fname = fname
    while os.path.exists(os.path.join(dest_dir, new_fname)):
        new_fname = f"{base}_{counter}{ext}"
        counter += 1

    return new_fname

