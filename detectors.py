from config import config
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

    def _make_batch(self, image: Image.Image):
        standard_im = self._preprocess_image(image).to(self.device)[None]

        batch = {"image": standard_im}
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

    def _detect(self, image: Image.Image, keypoint_count):
        batch = self._make_batch(image)

        detections = self.detector.detect(batch, keypoint_count)
        keypoints, confidences = detections["keypoints"], detections["confidence"]

        keypoints = keypoints.squeeze(0)
        confidences = confidences.squeeze(0)

        # confidence_distribution = get_confidence_distribution(torch.tensor(confidences))

        max_confidence, min_confidence = confidences.max(), confidences.min()
        confidences = (confidences - min_confidence) / (max_confidence - min_confidence)

        # confidence_distribution = get_confidence_distribution(confidences)

        return keypoints, confidences

    def _image_detect(self, kd: Keypoints):
        keypoint_count = config.dedode.image_keypoints_count
        keypoints, confidences = self._detect(kd.image, keypoint_count)

        kd.image_keypoints.normalised = keypoints
        kd.image_keypoints.confidences = confidences

    def _patches_detect(self, kd: Keypoints):
        keypoints_patches = []
        confidences_patches = []
        which_patch = []

        num_rows, num_cols = kd.patches_shape
        keypoint_count = config.dedode.patch_keypoints_count
        image_coords = kd.image_keypoints.as_image_coords()

        for i in range(num_rows):
            for j in range(num_cols):
                if not self._is_cell_empty(i, j, image_coords):
                    continue

                image = kd.patch_images[(i, j)]
                keypoints, confidences = self._detect(image, keypoint_count)

                keypoints_patches.append(keypoints)
                confidences_patches.append(confidences)
                which_patch.append((i, j))

        kd.patches_keypoints.normalised = torch.empty(0) if len(keypoints_patches) == 0 else torch.cat(keypoints_patches, dim=0)
        kd.patches_keypoints.confidences = torch.empty(0) if len(confidences_patches) == 0 else torch.cat(confidences_patches, dim=0)
        kd.patches_keypoints.which_patch = which_patch

    def extract_keypoints(self, image_names):
        for name in tqdm(image_names, desc="Extracting keypoints", ncols=100):
            # logger.info(f'Detector {name}')

            kd = Keypoints(name)

            self._image_detect(kd)
            self._patches_detect(kd)
            kd.save()
