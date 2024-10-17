from typing import List, Optional
from config import config
import cv2
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path


class ImageSoloData:
    def __init__(self, image_path):
        self.image_path: str = image_path
        self.image_name: str = Path(self.image_path).stem

        image = Image.open(image_path)
        image = image.resize(config.IMAGE_RESIZE)
        self.image: Image.Image = image

        self.keypoints = None
        self.keypoints_coords: Optional[List[cv2.KeyPoint]] = None
        self.confidences = None
        self.descriptions = None

    """
    Load & Save
    """

    def load_keypoints(self):
        filename = f"{self.image_name}_keypoints_{config.POSTFIX_DETECTOR_MODEL}_{config.POSTFIX_DATASET}.npy"
        keypoints_filepath = os.path.join(config.npy_dir_path, filename)
        assert os.path.exists(keypoints_filepath)
        self.keypoints = np.load(keypoints_filepath)

        filename = f"{self.image_name}_descriptions_{config.POSTFIX_DETECTOR_MODEL}_{config.POSTFIX_DATASET}.npy"
        descriptions_filepath = os.path.join(config.npy_dir_path, filename)
        assert os.path.exists(descriptions_filepath)
        self.descriptions = np.load(descriptions_filepath)

        self.keypoints_coords = [
            cv2.KeyPoint(
                int((x.item() + 1) * (self.image.width / 2)),
                int((y.item() + 1) * (self.image.height / 2)),
                1
            )
            for x, y in self.keypoints.squeeze(0)
        ]

    def save_keypoints(self):
        assert self.keypoints is not None and self.descriptions is not None

        keypoints_np = self.keypoints.cpu().numpy() if isinstance(self.keypoints, torch.Tensor) else self.keypoints
        descriptions_np = self.descriptions.cpu().numpy() if isinstance(self.descriptions, torch.Tensor) else self.descriptions

        filename = f"{self.image_name}_keypoints_{config.POSTFIX_DETECTOR_MODEL}_{config.POSTFIX_DATASET}.npy"
        np.save(os.path.join(config.npy_dir_path, filename), keypoints_np)

        filename = f"{self.image_name}_descriptions_{config.POSTFIX_DETECTOR_MODEL}_{config.POSTFIX_DATASET}.npy"
        np.save(os.path.join(config.npy_dir_path, filename), descriptions_np)


class ImagePairData:
    def __init__(self, a: ImageSoloData, b: ImageSoloData):
        self.a = a
        self.b = b

        self.left_matches = None
        self.right_matches = None

        self.left_matches_coords: Optional[List[cv2.KeyPoint]] = None
        self.right_matches_coords: Optional[List[cv2.KeyPoint]] = None

    """
    Load & Save
    """

    def load_matches(self):
        filename = f"{self.a.image_name}_{self.b.image_name}_matches_{config.POSTFIX_DETECTOR_MODEL}_{config.POSTFIX_MATCHER_MODEL}_{config.POSTFIX_DATASET}.npy"
        matches_filepath = os.path.join(config.npy_dir_path, filename)
        assert os.path.exists(matches_filepath)
        matches = np.load(matches_filepath)

        self.left_matches = matches[:, :2]
        self.right_matches = matches[:, 2:]

        self.left_matches_coords = [
            cv2.KeyPoint(int(x.item()), int(y.item()), 1.)
            for x, y in self.left_matches
        ]

        self.right_matches_coords = [
            cv2.KeyPoint(int(x.item()), int(y.item()), 1.)
            for x, y in self.right_matches
        ]

    def save_matches(self):
        filename = f"{self.a.image_name}_{self.b.image_name}_matches_{config.POSTFIX_DETECTOR_MODEL}_{config.POSTFIX_MATCHER_MODEL}_{config.POSTFIX_DATASET}.npy"
        matches_filepath = os.path.join(config.npy_dir_path, filename)

        left_matches_np = self.left_matches.cpu().numpy() if isinstance(self.left_matches, torch.Tensor) else self.left_matches
        right_matches_np = self.right_matches.cpu().numpy() if isinstance(self.right_matches, torch.Tensor) else self.right_matches

        matches = np.hstack([left_matches_np, right_matches_np])
        np.save(matches_filepath, matches)
