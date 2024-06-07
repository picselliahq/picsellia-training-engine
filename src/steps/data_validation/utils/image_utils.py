import os
from typing import List


def get_images_path_list(image_dir: str) -> List[str]:
    """
    Generates a list of all image file paths within a specified directory.

    Args:
        image_dir (str): The directory to search for image files.

    Returns:
        List[str]: A list containing the paths to all images found within the directory and its subdirectories.
    """
    images_path_list = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            images_path_list.append(os.path.join(root, file))
    return images_path_list
