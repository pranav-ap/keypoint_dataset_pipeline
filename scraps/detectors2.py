from typing import List
from config import config
from keypoint_dataset_pipeline.libs.DeDoDe.data_prep.prep_keypoints import image
from utils import get_best_device, logger
from ImageData import Keypoints
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from abc import ABC, abstractmethod
from tqdm import tqdm


def get_confidence_distribution(confidences):
    # Define the bins (0.0 to 1.0 with step of 0.1)
    bins = torch.arange(0, 1.1, 0.1).to(get_best_device())  # From 0 to 1 inclusive, with step 0.1

    # Digitize: find the index of the bin for each confidence value
    bin_indices = torch.bucketize(confidences, bins, right=True) - 1

    # Create an empty count tensor to hold the count of each bin
    counts = torch.zeros(len(bins) - 1, dtype=torch.int32)

    # Count the occurrences of each bin index
    for idx in bin_indices:
        if 0 <= idx < len(counts):
            counts[idx] += 1

    # Create a dictionary to map each bin range to its count
    distribution = {f"{bins[i]:.1f} - {bins[i+1]:.1f}": counts[i].item() for i in range(len(counts))}

    return distribution


class KeypointDetector(ABC):
    def __init__(self):
        self.device = get_best_device()

    @abstractmethod
    def extract_keypoints(self, image_names):
        pass


class DeDoDeDetector(KeypointDetector):
    def __init__(self):
        super().__init__()

        logger.info('Loading DeDoDeDetector')
        from DeDoDe import dedode_detector_L
        self.detector = dedode_detector_L(weights=None)
        logger.info('Loading DeDoDeDetector Done')

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

    def _make_batch(self, images: list[Image.Image]):
        # Preprocess each image and convert it to a tensor
        preprocessed_images = [self._preprocess_image(image).to(self.device) for image in images]

        # Stack all the preprocessed images into a single batch tensor
        batch = {"image": torch.stack(preprocessed_images)}

        return batch


    @staticmethod
    def _is_cell_empty(row, col, keypoints_coords) -> bool:
        patch_height, patch_width = config.image.patch_shape

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

    """
    Detect, Describe, Match
    """

    def _detect(self, images: List[Image.Image], keypoint_count):
        batch = self._make_batch(images)

        detections = self.detector.detect(batch, keypoint_count)
        keypoints_batch, confidences_batch = detections["keypoints"], detections["confidence"]

        return keypoints_batch, confidences_batch

    def _image_batch_detect(self, kds: List[Keypoints]):
        images = [kd.image for kd in kds]
        keypoint_count = config.dedode.image_keypoints_count
        keypoints_batch, confidences_batch = self._detect(images, keypoint_count)

        for kd, keypoints, confidences in zip(kds, keypoints_batch, confidences_batch):
            keypoints = keypoints.squeeze(0)
            confidences = confidences.squeeze(0)

            max_confidence, min_confidence = confidences.max(), confidences.min()
            confidences = (confidences - min_confidence) / (max_confidence - min_confidence)

            kd.image_keypoints.normalised = keypoints
            kd.image_keypoints.confidences = confidences

    def _patches_batch_detect(self, kds: List[Keypoints]):
        all_patches = []
        which_patch_indices = []

        # Iterate through each Keypoints object
        for kd in kds:
            num_rows, num_cols = kd.patches_shape
            image_coords = kd.image_keypoints.as_image_coords()

            for i in range(num_rows):
                for j in range(num_cols):
                    if self._is_cell_empty(i, j, image_coords):
                        continue

                    image = kd.patch_images[(i, j)]
                    all_patches.append(image)
                    which_patch_indices.append((kd, i, j))

        if all_patches:
            keypoint_count = config.dedode.patch_keypoints_count
            keypoints_batch, confidences_batch = self._detect(all_patches, keypoint_count)

            for (kd, i, j), keypoints, confidences in zip(which_patch_indices, keypoints_batch, confidences_batch):
                kd.patches_keypoints.normalised = keypoints
                kd.patches_keypoints.confidences = confidences

                if kd.patches_keypoints.which_patch is None:
                    kd.patches_keypoints.which_patch = []

                kd.patches_keypoints.which_patch.append((i, j))

        for kd in kds:
            if kd.patches_keypoints.normalised is None:
                kd.patches_keypoints.normalised = torch.empty(0)
            if kd.patches_keypoints.confidences is None:
                kd.patches_keypoints.confidences = torch.empty(0)

    def extract_keypoints(self, image_names):
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        batch_size = 4
        tot = len(image_names) // batch_size

        for image_batch in tqdm(chunks(image_names, batch_size), desc="Extracting keypoints", ncols=100, total=tot):
            keypoints_batch = [Keypoints(name) for name in image_batch]

            self._image_batch_detect(keypoints_batch)
            self._patches_batch_detect(keypoints_batch)

            for kd in keypoints_batch:
                kd.save()
