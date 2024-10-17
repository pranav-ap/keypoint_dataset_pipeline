from typing import List, Optional
from config import config
import cv2
import os
import torch
from PIL import Image
from pathlib import Path


def load_tensor(filename: str) -> torch.tensor:
    filepath: str = os.path.join(config.npy_dir_path, filename)
    assert os.path.exists(filepath)
    tensor: torch.tensor = torch.load(filepath)
    return tensor


def save_tensor(tensor: torch.tensor, filename: str):
    torch.save(tensor, os.path.join(config.npy_dir_path, filename))


class ImageSoloData:
    def __init__(self, image_path):
        self.image_path: str = image_path
        self.image_name: str = Path(self.image_path).stem

        image = Image.open(image_path)
        image = image.resize(config.IMAGE_RESIZE)
        self.image: Image.Image = image

        self.keypoints: Optional[torch.tensor] = None
        self.keypoints_coords: Optional[List[cv2.KeyPoint]] = None

        self.confidences: Optional[torch.tensor] = None
        self.descriptions: Optional[torch.tensor] = None

    def set_keypoints(self, keypoints: torch.tensor):
        self.keypoints = keypoints

        self.keypoints_coords = [
            cv2.KeyPoint(
                int((x.item() + 1) * (self.image.width / 2)),
                int((y.item() + 1) * (self.image.height / 2)),
                1
            )
            for x, y in self.keypoints.squeeze(0)
        ]

    """
    Load & Save
    """

    def load_keypoints(self):
        filename = f"{self.image_name}_keypoints_{config.POSTFIX_DETECTOR_MODEL}_{config.POSTFIX_DATASET}.pt"
        keypoints = load_tensor(filename)
        self.set_keypoints(keypoints)

        filename = f"{self.image_name}_descriptions_{config.POSTFIX_DETECTOR_MODEL}_{config.POSTFIX_DATASET}.pt"
        self.descriptions = load_tensor(filename)

        filename = f"{self.image_name}_confidences_{config.POSTFIX_DETECTOR_MODEL}_{config.POSTFIX_DATASET}.pt"
        self.confidences = load_tensor(filename)

    def save_keypoints(self):
        assert self.keypoints is not None
        assert self.descriptions is not None
        assert self.confidences is not None

        filename = f"{self.image_name}_keypoints_{config.POSTFIX_DETECTOR_MODEL}_{config.POSTFIX_DATASET}.pt"
        save_tensor(self.keypoints, filename)

        filename = f"{self.image_name}_descriptions_{config.POSTFIX_DETECTOR_MODEL}_{config.POSTFIX_DATASET}.pt"
        save_tensor(self.descriptions, filename)

        filename = f"{self.image_name}_confidences_{config.POSTFIX_DETECTOR_MODEL}_{config.POSTFIX_DATASET}.pt"
        save_tensor(self.confidences, filename)


class ImagePairData:
    def __init__(self, a: ImageSoloData, b: ImageSoloData):
        self.a = a
        self.b = b

        self.left_matches_coords: Optional[List[cv2.KeyPoint]] = None
        self.right_matches_coords: Optional[List[cv2.KeyPoint]] = None


class DSM_ImagePairData(ImagePairData):
    def __init__(self, a: ImageSoloData, b: ImageSoloData):
        super().__init__(a, b)

        self.left_matches: Optional[torch.tensor] = None
        self.right_matches: Optional[torch.tensor] = None

    def set_left_matches(self, left_matches):
        self.left_matches = left_matches
        self.left_matches_coords = [
            cv2.KeyPoint(int(x.item()), int(y.item()), 1.)
            for x, y in self.left_matches
        ]

    def set_right_matches(self, right_matches):
        self.right_matches = right_matches
        self.right_matches_coords = [
            cv2.KeyPoint(int(x.item()), int(y.item()), 1.)
            for x, y in self.right_matches
        ]

    """
    Load & Save
    """

    def load_matches(self):
        filename = f"{self.a.image_name}_{self.b.image_name}_matches_{config.POSTFIX_MATCHER_MODEL}_{config.POSTFIX_DATASET}.pt"
        matches = load_tensor(filename)

        self.set_left_matches(matches[:, :2])
        self.set_right_matches(matches[:, 2:])

    def save_matches(self):
        filename = f"{self.a.image_name}_{self.b.image_name}_matches_{config.POSTFIX_MATCHER_MODEL}_{config.POSTFIX_DATASET}.pt"
        matches = torch.cat([self.left_matches, self.right_matches], dim=1)
        save_tensor(matches, filename)


class RoMa_ImagePairData(ImagePairData):
    def __init__(self, a: ImageSoloData, b: ImageSoloData):
        super().__init__(a, b)

        self.warp: Optional[torch.tensor] = None
        self.pixel_coords: Optional[torch.tensor] = None

        self.certainty: Optional[torch.tensor] = None

    """
    Utils
    """

    def set_warp(self, warp):
        self.warp = warp
        self.pixel_coords = self._warp_to_pixel_coords()

    def set_left_matches_coords(self, left_matches_coords: List[cv2.KeyPoint]):
        self.left_matches_coords = left_matches_coords

    def set_right_matches_coords(self, right_matches_coords: List[cv2.KeyPoint]):
        self.right_matches_coords = right_matches_coords

    def _warp_to_pixel_coords(self):
        h1, w1 = self.a.image.height, self.a.image.width
        h2, w2 = self.b.image.height, self.b.image.width

        warp1 = self.warp[..., :2]
        warp1 = (
            torch.stack(
                (
                    w1 * (warp1[..., 0] + 1) / 2,
                    h1 * (warp1[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
        )

        warp2 = self.warp[..., 2:]
        warp2 = (
            torch.stack(
                (
                    w2 * (warp2[..., 0] + 1) / 2,
                    h2 * (warp2[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
        )

        return torch.cat((warp1, warp2), dim=-1)

    def get_random_reference_keypoints(self, confidence_threshold=0.6, num_points=5) -> List[cv2.KeyPoint]:
        assert self.certainty is not None

        # Create a binary mask where certainty is greater than the threshold
        mask = self.certainty > confidence_threshold

        # Get the coordinates of the points where mask is True
        y_coords, x_coords = torch.nonzero(mask, as_tuple=True)

        # If there are more points than required, randomly sample from them
        if len(y_coords) > num_points:
            indices = torch.randperm(len(y_coords))[:num_points]
        else:
            indices = torch.arange(len(y_coords))  # Take all if not enough points

        points = [
            cv2.KeyPoint(x_coords[i].item(), y_coords[i].item(), 1.)
            for i in indices
        ]

        return points

    def get_target_keypoints(self, reference_keypoints: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        target_keypoints = []
        # H, W = a.image.height, a.image.width

        for pt in reference_keypoints:
            x_a, y_a = pt.pt
            x_a, y_a = int(x_a), int(y_a)

            # w = warp[y_a, x_a]
            # A, B = self.model.to_pixel_coordinates(w, H, W, H, W)
            # x_b, y_b = B

            _, _, x_b, y_b = self.pixel_coords[y_a, x_a]
            x_b, y_b = int(x_b.item()), int(y_b.item())

            target_keypoints.append(cv2.KeyPoint(x_b, y_b, 1.))

        return target_keypoints

    """
    Load & Save
    """

    def load_warp_certainty(self):
        filename = f"{self.a.image_name}_{self.b.image_name}_warp_{config.POSTFIX_MATCHER_MODEL}_{config.POSTFIX_DATASET}.pt"
        warp = load_tensor(filename)
        self.set_warp(warp)

        filename = f"{self.a.image_name}_{self.b.image_name}_certainty_{config.POSTFIX_MATCHER_MODEL}_{config.POSTFIX_DATASET}.pt"
        self.certainty = load_tensor(filename)

    def save_warp_certainty(self):
        filename = f"{self.a.image_name}_{self.b.image_name}_warp_{config.POSTFIX_MATCHER_MODEL}_{config.POSTFIX_DATASET}.pt"
        save_tensor(self.warp, filename)

        filename = f"{self.a.image_name}_{self.b.image_name}_certainty_{config.POSTFIX_MATCHER_MODEL}_{config.POSTFIX_DATASET}.pt"
        save_tensor(self.certainty, filename)
