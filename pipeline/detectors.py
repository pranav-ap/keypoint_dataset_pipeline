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

    @staticmethod
    def _is_cell_empty(row, col, keypoints_coords) -> bool:
        patch_height, patch_width = config.image.patch_shape
        x_min, x_max = row * patch_width, (row + 1) * patch_width
        y_min, y_max = col * patch_height, (col + 1) * patch_height

        return all(not (x_min <= x < x_max and y_min <= y < y_max) for kp in keypoints_coords for x, y in [kp.pt])

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

    def _patches_detect(self, kds: List[Keypoints]):
        images, which_patch, which_image = [], [], []

        for kd in kds:
            num_rows, num_cols = kd.patches_shape
            image_coords = kd.image_keypoints.as_image_coords()

            for i in range(num_rows):
                for j in range(num_cols):
                    if not self._is_cell_empty(i, j, image_coords):
                        continue

                    images.append(kd.patch_images[(i, j)])
                    which_patch.append((i, j))
                    which_image.append(kd)

        keypoint_count = config.dedode.patch_keypoints_count
        keypoints, confidences = self._detect(images, keypoint_count)

        for kd, wp, keys, confs in zip(which_image, which_patch, keypoints, confidences):
            kd.patches_keypoints.normalised = torch.cat([kd.patches_keypoints.normalised, keys],
                                                        dim=0) if kd.patches_keypoints.normalised.numel() > 0 else keys
            kd.patches_keypoints.confidences = torch.cat([kd.patches_keypoints.confidences, confs],
                                                         dim=0) if kd.patches_keypoints.confidences.numel() > 0 else confs
            kd.patches_keypoints.which_patch.extend([wp] * keys.shape[0])

    def extract_keypoints(self, image_names):
        chunk_size = config.dedode.batch_size
        total_chunks = (len(image_names) + chunk_size - 1) // chunk_size

        for image_names_chunk in tqdm(chunk_iterable(image_names, chunk_size), total=total_chunks,
                                      desc="Extracting keypoints", ncols=100):
            kds = [Keypoints(name, self.data_store, is_filtered=False) for name in image_names_chunk]
            self._images_detect(kds)
            self._patches_detect(kds)

            for kd in kds:
                kd.save()
