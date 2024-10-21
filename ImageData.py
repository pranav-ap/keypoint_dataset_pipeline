from config import config
from utils import logger
import cv2
import os
import torch
import numpy as np
from PIL import Image
from skimage.util import view_as_blocks
from typing import List, Optional, Tuple, Dict


def load_tensor(filename: str) -> torch.tensor:
    filepath: str = os.path.join(config.paths[config.task].tensors_dir, filename)
    assert os.path.exists(filepath)
    tensor: torch.tensor = torch.load(filepath, weights_only=True)
    return tensor


def save_tensor(tensor: torch.tensor, filename: str):
    assert tensor is not None
    filepath = os.path.join(config.paths[config.task].tensors_dir, filename)
    torch.save(tensor, filepath)


class KeypointsData:
    def __init__(self, image_name):
        self.config = config

        self.image_path: str = f"{config.paths[config.task].images_dir}/{image_name}"

        image_name, _ = os.path.splitext(image_name)
        self.image_name: str = image_name

        self.image: Image.Image = self._init_image()

        grid_patches, grid_patches_shape = self._init_grid_patches()

        self.grid_patches: Dict[Tuple[int, int], Image.Image] = grid_patches
        self.grid_patches_shape: Tuple[int, int] = grid_patches_shape

        self.keypoints: Optional[torch.tensor] = None
        self.keypoints_coords: Optional[List[cv2.KeyPoint]] = None
        self.confidences: Optional[torch.tensor] = None

        self.keypoints_patches: Optional[torch.tensor] = None
        self.keypoints_patches_coords: Optional[List[cv2.KeyPoint]] = None
        self.confidences_patches: Optional[torch.tensor] = None

    def _init_image(self):
        image = Image.open(self.image_path)
        x, y = self.config.image.resize
        image = image.resize((x, y))
        image = image.convert('RGB')
        return image

    def _init_grid_patches(self):
        assert self.image is not None

        # image_np.shape (896, 896, 3) example
        image_np = np.array(self.image)
        # view.shape (7, 7, 1, 128, 128, 3)
        x, y, z = self.config.image.patch_shape
        view = view_as_blocks(image_np, (x, y, z))
        # view.shape (7, 7, 128, 128, 3)
        view = view.squeeze()

        # grid_patches shape (7, 7)
        grid_patches = {}
        grid_patches_shape = view.shape[0], view.shape[1]  # rows, cols

        for i in range(view.shape[0]):
            for j in range(view.shape[1]):  # Loop through the horizontal patches
                patch = Image.fromarray(view[i, j])
                grid_patches[(i, j)] = patch

        return grid_patches, grid_patches_shape

    def init_keypoints(self, keypoints: torch.tensor):
        assert keypoints is not None

        self.keypoints = keypoints

        self.keypoints_coords = [
            cv2.KeyPoint(
                int((x.item() + 1) * (self.image.width / 2)),
                int((y.item() + 1) * (self.image.height / 2)),
                1
            )
            for x, y in self.keypoints
        ]

    def _init_keypoints_patches(self, keypoints_patches: torch.tensor):
        assert keypoints_patches is not None

        self.keypoints_patches = keypoints_patches

        self.keypoints_patches_coords = [
            cv2.KeyPoint(
                int((x.item() + 1) * (self.image.width / 2)),
                int((y.item() + 1) * (self.image.height / 2)),
                1
            )
            for x, y in self.keypoints_patches
        ]

    """
    Load & Save
    """

    def load(self):
        if self.keypoints is None or self.keypoints_coords is None:
            filename = f"{self.image_name}_keypoints.pt"
            keypoints = load_tensor(filename)
            self.init_keypoints(keypoints)

        if self.confidences is None:
            filename = f"{self.image_name}_confidences.pt"
            self.confidences = load_tensor(filename)

        if self.keypoints_patches is None or self.keypoints_patches_coords is None:
            filename = f"{self.image_name}_keypoints_patches.pt"
            keypoints_patches = load_tensor(filename)
            self._init_keypoints_patches(keypoints_patches)

        if self.confidences_patches is None:
            filename = f"{self.image_name}_confidences_patches.pt"
            self.confidences_patches = load_tensor(filename)

    def save(self):
        filename = f"{self.image_name}_keypoints.pt"
        save_tensor(self.keypoints, filename)

        filename = f"{self.image_name}_confidences.pt"
        save_tensor(self.confidences, filename)

        filename = f"{self.image_name}_keypoints_patches.pt"
        save_tensor(self.keypoints_patches, filename)

        filename = f"{self.image_name}_confidences_patches.pt"
        save_tensor(self.confidences_patches, filename)


class MatchesData:
    def __init__(self, a: KeypointsData, b: KeypointsData):
        self.config = config

        self.a = a
        self.b = b

        self.warp: Optional[torch.tensor] = None
        self.pixel_coords: Optional[torch.tensor] = None
        self.certainty: Optional[torch.tensor] = None

        self.left_matches_coords_filtered: Optional[List[cv2.KeyPoint]] = None
        self.right_matches_coords_filtered: Optional[List[cv2.KeyPoint]] = None

    @staticmethod
    def load_from_names(name_a, name_b, load_filtered_matches=False):
        a = KeypointsData(name_a)
        a.load()

        b = KeypointsData(name_b)
        b.load()

        pair = MatchesData(a, b)
        pair.load()

        if load_filtered_matches:
            pair.load_filtered_matches()

        return pair

    """
    Utils
    """

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
                axis=-1
            )
        )

        warp2 = self.warp[..., 2:]
        warp2 = (
            torch.stack(
                (
                    w2 * (warp2[..., 0] + 1) / 2,
                    h2 * (warp2[..., 1] + 1) / 2,
                ),
                axis=-1
            )
        )

        return torch.cat((warp1, warp2), dim=-1)

    def set_warp(self, warp):
        self.warp = warp
        self.pixel_coords = self._warp_to_pixel_coords()

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
            cv2.KeyPoint(int(x_coords[i].item()), int(y_coords[i].item()), 1.)
            for i in indices
        ]

        return points

    def get_good_matches(self, reference_keypoints: List[cv2.KeyPoint], confidence_threshold=0.6) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint]]:
        target_keypoints = []
        accepted_reference_keypoints = []

        for pt in reference_keypoints:
            x_a, y_a = pt.pt
            x_a, y_a = int(x_a), int(y_a)

            conf = self.certainty[y_a, x_a]
            if conf <= confidence_threshold:
                continue

            _, _, x_b, y_b = self.pixel_coords[y_a, x_a]
            x_b, y_b = int(x_b.item()), int(y_b.item())

            accepted_reference_keypoints.append(pt)
            target_keypoints.append(cv2.KeyPoint(x_b, y_b, 1.))

        return accepted_reference_keypoints, target_keypoints

    """
    Load & Save
    """

    def load(self):
        if self.warp is None or self.pixel_coords is None:
            filename = f"{self.a.image_name}_{self.b.image_name}_warp.pt"
            warp = load_tensor(filename)
            self.set_warp(warp)

        if self.certainty is None:
            filename = f"{self.a.image_name}_{self.b.image_name}_certainty.pt"
            self.certainty = load_tensor(filename)

    def save(self):
        filename = f"{self.a.image_name}_{self.b.image_name}_warp.pt"
        save_tensor(self.warp, filename)

        filename = f"{self.a.image_name}_{self.b.image_name}_certainty.pt"
        save_tensor(self.certainty, filename)

    def load_filtered_matches(self):
        if self.left_matches_coords_filtered is None or self.right_matches_coords_filtered is None:
            filename = f"{self.a.image_name}_{self.b.image_name}_matches.pt"
            matches = load_tensor(filename)

            logger.debug(f'matches.shape {matches.shape}')

            self.left_matches_coords_filtered = [
                cv2.KeyPoint(int(x), int(y), 1.)
                for x, y in matches[:, :2]
            ]

            self.right_matches_coords_filtered = [
                cv2.KeyPoint(int(x), int(y), 1.)
                for x, y in matches[:, 2:]
            ]

            logger.debug(f'len(self.left_matches_coords_filtered) {len(self.left_matches_coords_filtered)}')

    def save_filtered_matches(self):
        assert self.left_matches_coords_filtered is not None and self.right_matches_coords_filtered is not None

        filename = f"{self.a.image_name}_{self.b.image_name}_matches.pt"

        left_matches_coords_filtered = torch.tensor([kp.pt for kp in self.left_matches_coords_filtered])
        right_matches_coords_filtered = torch.tensor([kp.pt for kp in self.right_matches_coords_filtered])

        matches = torch.cat([left_matches_coords_filtered, right_matches_coords_filtered], dim=1)
        save_tensor(matches, filename)
