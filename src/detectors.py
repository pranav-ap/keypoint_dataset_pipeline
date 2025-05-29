from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from config import config
from utils import chunk_iterable, get_best_device, logger
from .ImageData import Keypoints


class KeypointDetector(ABC):
    device = get_best_device()

    def __init__(self, data_store):
        self.data_store = data_store

    @abstractmethod
    def extract_keypoints(self, image_names):
        pass


class DeDoDeDetector(KeypointDetector):
    def __init__(self, data_store):
        super().__init__(data_store)

        logger.info('Loading DeDoDeDetector')
        from DeDoDe import dedode_detector_L
        self.detector = dedode_detector_L(weights=None, remove_borders=True)
        logger.info('Loading DeDoDeDetector Done')

        self.normalizer = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _preprocess_image(self, image: Image.Image):
        standard_im = np.array(image) / 255.0

        if standard_im.ndim == 2:
            standard_im = np.stack([standard_im] * 3, axis=0)  # (3, H, W)
        else:
            standard_im = np.transpose(standard_im, (2, 0, 1))  # (3, H, W)

        x = torch.from_numpy(standard_im).float().to(self.device)
        return self.normalizer(x)

    @torch.no_grad()
    def _detect(self, images: List[Image.Image], keypoint_count):
        image_processed = [self._preprocess_image(image).to(self.device) for image in images]
        batch = {"image": torch.stack(image_processed)}

        detections = self.detector.detect(batch, keypoint_count)

        return detections["keypoints"], detections["confidence"]

    def _images_detect(self, kds: List[Keypoints]):
        keypoint_count = config.dedode.image_keypoints_count
        images = [kd.image for kd in kds]
        keypoints_batch, confidences_batch = self._detect(images, keypoint_count)

        for kd, keypoints, confidences in zip(kds, keypoints_batch, confidences_batch):
            kd.image_keypoints.normalised = keypoints
            kd.image_keypoints.confidences = confidences

    def extract_keypoints(self, image_names):
        chunk_size = config.dedode.batch_size
        total_chunks = (len(image_names) + chunk_size - 1) // chunk_size

        for image_names_chunk in tqdm(
                chunk_iterable(image_names, chunk_size), 
                total=total_chunks,
                desc="Extracting keypoints", 
                ncols=100,
            ):
            kds = [Keypoints(name, self.data_store, is_filtered=False) for name in image_names_chunk]
            self._images_detect(kds)

            for kd in kds:
                kd.save()
