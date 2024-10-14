from utils.logger import logger
from utils import get_best_device
from ImageData import ImageSoloData
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from DeDoDe import dedode_detector_L, dedode_descriptor_G
from pathlib import Path
from typing import Optional


class DataPipeline:
    def __init__(self, num_dedode_keypoints=1000):
        self.detector = dedode_detector_L(weights=None)
        self.descriptor = dedode_descriptor_G(weights=None, dinov2_weights=None)

        self.normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = get_best_device()

        self.num_dedode_keypoints = num_dedode_keypoints

    """
    Detect & Describe
    """

    def preprocess_dedode_image(self, image_data: ImageSoloData):
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

    def detect_describe(self, image_path) -> ImageSoloData:
        """
        Returns an Image Data object that contains keypoints and descriptors
        """

        image_data = ImageSoloData(image_path, resize=(784, 784))
        standard_im = self.preprocess_dedode_image(image_data).to(self.device)[None]  # Add batch dimension
        batch = {"image": standard_im}

        detections = self.detector.detect(batch, self.num_dedode_keypoints)
        image_data.keypoints = detections["keypoints"]
        image_data.confidences = detections["confidence"]

        descriptions = self.descriptor.describe_keypoints(batch, image_data.keypoints)
        image_data.descriptions = descriptions["descriptions"]

        return image_data


class DataPipelineRunner:
    def __init__(self):
        pass

