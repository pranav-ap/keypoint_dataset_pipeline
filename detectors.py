from config import config
from ImageData import KeypointsData
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from abc import ABC, abstractmethod
from typing import List
from rich.progress import Progress


class KeypointDetector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def extract_keypoints(self, image_names):
        pass


class DeDoDeDetector(KeypointDetector):
    def __init__(self):
        super().__init__()

        from DeDoDe import dedode_detector_L
        self.detector = dedode_detector_L(weights=None)

        self.normalizer = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    """
    Utils
    """

    def _preprocess_image(self, image: Image.Image):
        standard_im = np.array(image) / 255.0

        # Convert grayscale to 3-channel if needed
        if standard_im.ndim == 2:
            standard_im = np.stack([standard_im] * 3, axis=0)  # (3, H, W)
        else:
            standard_im = np.transpose(standard_im, (2, 0, 1))  # (3, H, W)

        standard_im = self.normalizer(torch.from_numpy(standard_im)).float()

        return standard_im

    def _make_batch(self, image: Image.Image):
        standard_im = self._preprocess_image(image).to(config.device)[None]
        batch = {"image": standard_im}
        return batch

    """
    Detect, Describe, Match
    """

    def _detect(self, image: Image.Image, keypoint_count):
        batch = self._make_batch(image)

        detections = self.detector.detect(batch, keypoint_count)
        keypoints, confidences = detections["keypoints"], detections["confidence"]

        return keypoints, confidences

    def _image_detect(self, kd: KeypointsData):
        keypoint_count = config.dedode.image_keypoints_count

        keypoints, confidences = self._detect(
            kd.image,
            keypoint_count
        )

        kd.init_keypoints(keypoints)
        kd.confidences = confidences

    def _patch_detect(self, image: Image.Image, row, col):
        keypoint_count = config.dedode.patch_keypoints_count

        keypoints, confidences = self._detect(
            image,
            keypoint_count
        )

        patch_height, patch_width, _ = config.image.patch_shape
        keypoints_coords: List[cv2.KeyPoint] = []

        for x, y in keypoints.squeeze(0):
            x = int((x.item() + 1) * (patch_width / 2))
            y = int((y.item() + 1) * (patch_height / 2))

            global_x = x + row * patch_width
            global_y = y + col * patch_height

            kp = cv2.KeyPoint(global_x, global_y, 1)

            keypoints_coords.append(kp)

        return keypoints, keypoints_coords, confidences

    @staticmethod
    def _is_cell_empty(row, col, keypoints_coords) -> bool:
        patch_height, patch_width, _ = config.image.patch_shape

        # Define the bounds of the cell
        x_min = row * patch_width
        x_max = (row + 1) * patch_width
        y_min = col * patch_height
        y_max = (col + 1) * patch_height

        # Check if any point falls inside this cell
        for kp in keypoints_coords:
            x, y = kp.pt
            if x_min <= x < x_max and y_min <= y < y_max:
                # Cell is not empty, a point is inside
                return False

        # No points found, cell is empty
        return True

    def _patches_detect(self, kd: KeypointsData):
        keypoints_patches = []
        keypoints_coords_patches = []
        confidences_patches = []

        num_rows, num_cols = kd.grid_patches_shape

        for i in range(num_rows):
            for j in range(num_cols):
                if not self._is_cell_empty(i, j, kd.keypoints_coords):
                    continue

                patch = kd.grid_patches[(i, j)]
                package = self._patch_detect(patch, i, j)
                keypoints, keypoints_coords, confidences = package

                keypoints_patches.append(keypoints)
                keypoints_coords_patches.extend(keypoints_coords)
                confidences_patches.append(confidences)

        kd.keypoints_patches = torch.cat(keypoints_patches, dim=0)
        kd.keypoints_patches_coords = keypoints_coords_patches
        kd.confidences_patches = torch.cat(confidences_patches, dim=0)

    def extract_keypoints(self, image_names):
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Extracting keypoints...",
                total=len(image_names)
            )

            for name in image_names:
                kd = KeypointsData(name)

                self._image_detect(kd)
                self._patches_detect(kd)
                kd.save()

                progress.advance(task)
