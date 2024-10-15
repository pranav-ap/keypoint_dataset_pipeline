import cv2
import random
from config import config
from ImageData import ImageSoloData
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from typing import Optional
from abc import ABC, abstractmethod


class KeypointDetector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def extract_keypoints(self, image_names):
        pass


class DeDoDeDetector(KeypointDetector):
    def __init__(self):
        super().__init__()

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

    def _preprocess_image(self, image_data: ImageSoloData):
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
    Visualization
    """

    @staticmethod
    def show_keypoints(path_im, num_points=10):
        im = ImageSoloData(path_im, resize=config.IMAGE_RESIZE)
        im.load_keypoints()

        width, height = im.image.size
        image = np.array(im.image)

        keypoints = [
            cv2.KeyPoint((x + 1) * (width/2), (y+1) * (height/2), 1.)
            for x, y in im.keypoints.squeeze(0)
        ]

        num_points = len(keypoints) if num_points is None else min(num_points, len(keypoints))
        keypoints = random.sample(keypoints, num_points)

        image_vis = cv2.drawKeypoints(
            image,
            keypoints,
            None
        )

        image_vis = Image.fromarray(image_vis)

        return image_vis

    """
    Detect, Describe, Match
    """

    def _detect_describe(self, image_data: ImageSoloData):
        """
        Returns an Image Data object that contains keypoints and descriptors
        """

        standard_im = self._preprocess_image(image_data).to(config.device)[None]
        batch = {"image": standard_im}

        detections = self.detector.detect(batch, config.num_keypoints_to_detect)
        image_data.keypoints = detections["keypoints"]
        image_data.confidences = detections["confidence"]

        descriptions = self.descriptor.describe_keypoints(batch, image_data.keypoints)
        image_data.descriptions = descriptions["descriptions"]

    def extract_keypoints(self, image_names):
        a: Optional[ImageSoloData] = None
        b: Optional[ImageSoloData] = None

        for index in range(len(image_names) - 1):
            path_a = f"{config.images_dir_path}/{image_names[index]}"
            path_b = f"{config.images_dir_path}/{image_names[index + 1]}"

            if a is None:
                a = ImageSoloData(path_a, resize=config.IMAGE_RESIZE)
                self._detect_describe(a)

            b = ImageSoloData(path_b, resize=config.IMAGE_RESIZE)
            self._detect_describe(b)

            a.save_keypoints()
            a = b

        if b:
            b.save_keypoints()
