from utils.logger import logger
from config import config
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path


class ImageSoloData:
    def __init__(self, image_path, resize=None):
        self.image_path: str = image_path
        self.image_name: str = Path(self.image_path).stem

        self.image: Image.Image = Image.open(image_path)

        if resize:
            self.image = self.image.resize(resize)

        W, H = self.image.size
        self.W: int = W
        self.H: int = H

        self.keypoints = None
        self.confidences = None
        self.descriptions = None

    """
    Load & Save
    """

    def load_keypoints(self):
        POSTFIX_FILE = f'{config.POSTFIX_MODEL}_{config.POSTFIX_DATASET}'

        keypoints_filepath = os.path.join(config.npy_dir_path, f"{self.image_name}_keypoints_{POSTFIX_FILE}.npy")
        assert os.path.exists(keypoints_filepath)
        self.keypoints = np.load(keypoints_filepath)
        logger.debug(f'LOAD self.keypoints.shape {self.keypoints.shape}')

        descriptions_filepath = os.path.join(config.npy_dir_path, f"{self.image_name}_descriptions_{POSTFIX_FILE}.npy")
        assert os.path.exists(descriptions_filepath)
        self.descriptions = np.load(descriptions_filepath)
        logger.debug(f'LOAD self.descriptions.shape {self.descriptions.shape}')

    def save_keypoints(self):
        assert self.keypoints is not None and self.descriptions is not None

        keypoints_np = self.keypoints.cpu().numpy() if isinstance(self.keypoints, torch.Tensor) else self.keypoints
        descriptions_np = self.descriptions.cpu().numpy() if isinstance(self.descriptions, torch.Tensor) else self.descriptions

        logger.debug(f'SAVE keypoints_np.shape {keypoints_np.shape}')
        logger.debug(f'SAVE descriptions_np.shape {descriptions_np.shape}')

        POSTFIX_FILE = f'{config.POSTFIX_MODEL}_{config.POSTFIX_DATASET}'

        np.save(os.path.join(config.npy_dir_path, f"{self.image_name}_keypoints_{POSTFIX_FILE}.npy"), keypoints_np)
        np.save(os.path.join(config.npy_dir_path, f"{self.image_name}_descriptions_{POSTFIX_FILE}.npy"), descriptions_np)


class ImagePairData:
    def __init__(self, a: ImageSoloData, b: ImageSoloData):
        self.a = a
        self.b = b

        self.left_matches = None
        self.right_matches = None

    """
    Load & Save
    """

    def load_matches(self):
        POSTFIX_FILE = f'{config.POSTFIX_MODEL}_{config.POSTFIX_DATASET}'
        filename = f'{self.a.image_name}_{self.b.image_name}_matches_{POSTFIX_FILE}.npy'

        matches_filepath = os.path.join(config.npy_dir_path, filename)
        assert os.path.exists(matches_filepath)

        matches = np.load(matches_filepath)
        logger.debug(f'LOAD matches.shape {matches.shape}')

        self.left_matches = matches[:, :2]
        self.right_matches = matches[:, 2:]

    def save_matches(self):
        POSTFIX_FILE = f'{config.POSTFIX_MODEL}_{config.POSTFIX_DATASET}'
        filename = f'{self.a.image_name}_{self.b.image_name}_matches_{POSTFIX_FILE}.npy'

        matches_filepath = os.path.join(config.npy_dir_path, filename)

        left_matches_np = self.left_matches.cpu().numpy() if isinstance(self.left_matches, torch.Tensor) else self.left_matches
        right_matches_np = self.right_matches.cpu().numpy() if isinstance(self.right_matches, torch.Tensor) else self.right_matches

        logger.debug(f'SAVE left_matches_np.shape {left_matches_np.shape}')
        logger.debug(f'SAVE right_matches_np.shape {right_matches_np.shape}')

        matches = np.hstack([left_matches_np, right_matches_np])
        np.save(matches_filepath, matches)
