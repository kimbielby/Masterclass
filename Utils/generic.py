import os
import shutil
import cv2

def save_img_as(fname, img):
    cv2.imwrite(filename=fname, img=img)

def get_unique_filename(dest_dir, fname):
    """
    When moving a file to a new directory, it checks there is not already a file in
    there with the same name. If there is, it adds a number to the end of the filename
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

def move_files(base_dir, sub_dir_list, dest_dir):
    """
    Moves all files in sub_dir_list to dest_dir and deletes folders once emptied.
    :param base_dir: Where the folders with files currently live, and where to create
        destination directory
    :param sub_dir_list: List of folders where the files currently live
    :param dest_dir: Folder where the files will be moved to
    """
    # Create destination directory if it does not already exist
    os.makedirs(dest_dir, exist_ok=True)

    # Loop through each directory that holds the files and move them
    for folder in sub_dir_list:
        src_path = os.path.join(base_dir, folder)
        for filename in os.listdir(src_path):
            full_path = os.path.join(src_path, filename)
            if os.path.isfile(full_path):
                unique_fname = get_unique_filename(dest_dir=dest_dir, fname=filename)
                shutil.move(full_path, os.path.join(dest_dir, unique_fname))

        # Delete folder once it's empty
        if not os.listdir(src_path):
            os.rmdir(src_path)

def rename_files_in_folder(folder, new_name_s, new_name_e):
    """
    Renames all files in a given directory
    :param folder: When the files live
    :param new_name_s: Start of the new name
    :param new_name_e: End of the new name
    """
    # Get all files in the folder
    files = [f for f in os.listdir(folder)]

    # Find how many zeros for filename padding
    num_files = len(files)
    padding = len(str(num_files - 1))

    # Rename each image
    for idx, old_name in enumerate(files):
        ext = os.path.splitext(old_name)[1]
        # New name example: img_0012_gt.jpg
        new_name = f"{new_name_s}{idx:0{padding}d}{new_name_e}{ext}"
        old_path = os.path.join(folder, old_name)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)

def rename_frames(folder, new_fname):
    """
    Renames the frames (images) in the given folder
    :param folder: The gt or spill folder where the frames to be renamed live
    :param new_fname: The new base filename, e.g. gt_kim_walking2
    """
    for filename in os.listdir(folder):
        # Get the number part of the og filename
        img_num = filename.split('.')[1]
        # Use it in the new filename
        new_filename = f"{new_fname}_{img_num}.png"
        # Specify og path and new path
        og_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_filename)
        # Rename the file
        os.rename(og_path, new_path)
        print(f"Renamed {filename} to {new_filename}")

