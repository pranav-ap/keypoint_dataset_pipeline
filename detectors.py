from utils.logger import logger
from config import config
from utils import get_best_device
from ImageData import ImageSoloData
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from typing import Optional
from abc import ABC, abstractmethod


class KeypointDetector(ABC):
    def __init__(self, num_keypoints):
        self.device = get_best_device()
        self.num_keypoints = num_keypoints

    @abstractmethod
    def extract_keypoints(self):
        pass


class DeDoDeDetector(KeypointDetector):
    def __init__(self, image_names, num_keypoints=1000):
        super().__init__(num_keypoints)

        self.image_names = image_names

        from DeDoDe import dedode_detector_L, dedode_descriptor_G
        self.detector = dedode_detector_L(weights=None)
        self.descriptor = dedode_descriptor_G(weights=None, dinov2_weights=None)

        self.normalizer = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    """
    Utils
    """

    def preprocess_image(self, image_data: ImageSoloData):
        # Convert grayscale to RGB if necessary
        if image_data.image.mode != 'RGB':
            image_data.image = image_data.image.convert('RGB')

        standard_im = np.array(image_data.image) / 255.0

        # Convert grayscale to 3-channel if needed
        if standard_im.ndim == 2:
            standard_im = np.stack([standard_im] * 3, axis=0)  # (3, H, W)
        else:
            standard_im = np.transpose(standard_im, (2, 0, 1))  # (3, H, W)

        standard_im = self.normalizer(torch.from_numpy(standard_im)).float()

        return standard_im

    """
    Detect, Describe, Match
    """

    def detect_describe(self, image_data: ImageSoloData):
        """
        Returns an Image Data object that contains keypoints and descriptors
        """

        standard_im = self.preprocess_image(image_data).to(self.device)[None]
        batch = {"image": standard_im}

        detections = self.detector.detect(batch, self.num_keypoints)
        image_data.keypoints = detections["keypoints"]
        image_data.confidences = detections["confidence"]

        descriptions = self.descriptor.describe_keypoints(batch, image_data.keypoints)
        image_data.descriptions = descriptions["descriptions"]

    def extract_keypoints(self):
        a: Optional[ImageSoloData] = None
        b: Optional[ImageSoloData] = None

        IMAGE_RESIZE = (784, 784)
        FILE_POSTFIX = f'{config.POSTFIX_DEDODE}_{config.POSTFIX_EUROC}'

        for index in range(len(self.image_names) - 1):
            path_a = f"{config.images_dir_path}/{self.image_names[index]}"
            path_b = f"{config.images_dir_path}/{self.image_names[index + 1]}"

            if a is None:
                a = ImageSoloData(path_a, resize=IMAGE_RESIZE, file_postfix=FILE_POSTFIX)
                self.detect_describe(a)

            b = ImageSoloData(path_b, resize=IMAGE_RESIZE, file_postfix=FILE_POSTFIX)
            self.detect_describe(b)

            a.save_keypoints()
            a = b

        if b:
            b.save_keypoints()
