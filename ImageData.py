from config import config
import os
import numpy as np
from PIL import Image
from pathlib import Path


class ImageSoloData:
    def __init__(self, image_path, resize=None, file_postfix=None):
        self.image_path: str = image_path
        self.image_name: str = Path(self.image_path).stem

        self.image: Image.Image = Image.open(image_path)

        if resize:
            self.image = self.image.resize(resize)

        W, H = self.image.size
        self.W: int = W
        self.H: int = H

        self.file_postfix = file_postfix

        self.keypoints = None
        self.confidences = None
        self.descriptions = None

    """
    Load & Save
    """

    def load_keypoints(self):
        keypoints_filepath = os.path.join(config.npy_dir_path, f"{self.image_name}_keypoints_{self.file_postfix}.npy")
        assert os.path.exists(keypoints_filepath)
        self.keypoints = np.load(keypoints_filepath)

        descriptions_filepath = os.path.join(config.npy_dir_path, f"{self.image_name}_descriptions_{self.file_postfix}.npy")
        assert os.path.exists(descriptions_filepath)
        self.descriptions = np.load(descriptions_filepath)

    def save_keypoints(self):
        assert self.keypoints is not None and self.descriptions is not None

        keypoints_np = self.keypoints.cpu().numpy()
        descriptions_np = self.descriptions.cpu().numpy()

        np.save(os.path.join(config.npy_dir_path, f"{self.image_name}_keypoints_{self.file_postfix}.npy"), keypoints_np)
        np.save(os.path.join(config.npy_dir_path, f"{self.image_name}_descriptions_{self.file_postfix}.npy"), descriptions_np)


class ImagePairData:
    def __init__(self, a: ImageSoloData, b: ImageSoloData, file_postfix: str):
        self.a = a
        self.b = b

        self.file_postfix = file_postfix
        self.filename = f'{a.image_name}_{b.image_name}_matches_{self.file_postfix}.npy'

        self.left_matches = None
        self.right_matches = None

    """
    Load & Save
    """

    def load_matches(self):
        matches_filepath = os.path.join(config.npy_dir_path, self.filename)
        assert os.path.exists(matches_filepath)

        matches = np.load(matches_filepath)

        self.left_matches = matches[:, :2]
        self.right_matches = matches[:, 2:]

    def save_matches(self):
        matches_filepath = os.path.join(config.npy_dir_path, self.filename)
        matches = np.hstack([self.left_matches, self.right_matches])

        np.save(matches_filepath, matches)
