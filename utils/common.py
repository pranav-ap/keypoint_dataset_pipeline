import os
import shutil
import zipfile
from itertools import islice

import torch


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


def make_clear_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Remove all contents of the directory
        shutil.rmtree(directory_path)

    # Recreate the directory
    os.makedirs(directory_path, exist_ok=True)


def zip_folder(folder_path, output_zip, wild):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(wild):
                    file_path = os.path.join(root, file)
                    # noinspection PyTypeChecker
                    zip_file.write(file_path, os.path.relpath(file_path, folder_path))
