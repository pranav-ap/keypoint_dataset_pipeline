import os
import zipfile
import shutil
import torch
from PIL import Image
from itertools import islice


def chunk_iterable(iterable, chunk_size):
    iterator = iter(iterable)
    for first in iterator:
        yield [first] + list(islice(iterator, chunk_size - 1))


def get_best_device(verbose=False):
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    if verbose:
        print(f"Fastest device found is: {device}")

    return device


def list_files_in_folder(folder_path):
    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)

    return file_paths


def make_clear_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Remove all contents of the directory
        shutil.rmtree(directory_path)

    # Recreate the directory (optional, if you want to keep the directory itself)
    os.makedirs(directory_path, exist_ok=True)


def resize_image_to_nearest_multiple(image: Image.Image, patch_height: int, patch_width: int, scale_factor: float = 1.0):
    """
    image = Image.open(image_path)
    resized_image = resize_image_to_nearest_multiple(image, 14, 14, scale_factor=0.25)  # Scale down by half
    resized_image.size
    """
    width, height = image.size

    # Apply scale factor to the original dimensions
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)

    # Calculate the new dimensions rounded to the nearest multiple of patch height and width
    new_width = round(scaled_width / patch_width) * patch_width
    new_height = round(scaled_height / patch_height) * patch_height

    # Resize the image to the new dimensions
    resized_image = image.resize((new_width, new_height))

    return resized_image


def zip_folder(folder_path, output_zip, wild):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(wild):
                    file_path = os.path.join(root, file)
                    # noinspection PyTypeChecker
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))

