from config import config
from utils import logger, get_best_device
import cv2
import os
import torch
import numpy as np
from PIL import Image
from skimage.util import view_as_blocks
from typing import List, Optional, Tuple, Dict
from abc import ABC


def load_tensor(filename: str) -> torch.tensor:
    filepath: str = os.path.join(config.paths[config.task.name].tensors_dir, filename)
    assert os.path.exists(filepath)
    tensor: torch.tensor = torch.load(filepath, weights_only=True)
    return tensor


def save_tensor(tensor: torch.tensor, filename: str):
    assert tensor is not None
    filepath = os.path.join(config.paths[config.task.name].tensors_dir, filename)
    torch.save(tensor, filepath)


"""
KEYPOINTS
"""


class _Keypoints(ABC):
    device = get_best_device()

    def __init__(self, image_name, is_filtered):
        self.image_name: str = image_name
        self.is_filtered: bool = is_filtered

        self.normalised: torch.Tensor = torch.empty(0, device=self.device)
        self.confidences: torch.Tensor = torch.empty(0, device=self.device)


class ImageKeypoints(_Keypoints):
    def __init__(self, image_name, is_filtered=False):
        super().__init__(image_name, is_filtered)

    def as_image_coords(self) -> List[cv2.KeyPoint]:
        w, h = config.image.image_shape

        if self.normalised.numel() == 0:
            return []

        coords = [
            cv2.KeyPoint(
                int((x.item() + 1) * (w / 2)),
                int((y.item() + 1) * (h / 2)),
                1
            )
            for x, y in self.normalised
        ]

        return coords

    def load(self):
        filename = f"{self.image_name}_keypoints_normalised.pt" if not self.is_filtered else f"{self.image_name}_keypoints_normalised_filtered.pt"
        self.normalised = load_tensor(filename)

        filename = f"{self.image_name}_confidences.pt" if not self.is_filtered else f"{self.image_name}_confidences_filtered.pt"
        self.confidences = load_tensor(filename)

    def save(self):
        filename = f"{self.image_name}_keypoints_normalised.pt" if not self.is_filtered else f"{self.image_name}_keypoints_normalised_filtered.pt"
        save_tensor(self.normalised, filename)

        filename = f"{self.image_name}_confidences.pt" if not self.is_filtered else f"{self.image_name}_confidences_filtered.pt"
        save_tensor(self.confidences, filename)


class PatchesKeypoints(_Keypoints):
    def __init__(self, image_name, is_filtered=False):
        super().__init__(image_name, is_filtered)
        self.which_patch: List[Tuple[int, int]] = []

    def as_image_coords(self) -> List[cv2.KeyPoint]:
        patch_height, patch_width = config.image.patch_shape
        coords = []

        for (x, y), (row, col) in zip(self.normalised, self.which_patch):
            x = int((x.item() + 1) * (patch_width / 2))
            y = int((y.item() + 1) * (patch_height / 2))

            global_x = x + row * patch_width
            global_y = y + col * patch_height

            kp = cv2.KeyPoint(global_x, global_y, 1)
            coords.append(kp)

        return coords

    def load(self):
        filename = f"{self.image_name}_keypoints_normalised_patches.pt" if not self.is_filtered else f"{self.image_name}_keypoints_normalised_patches_filtered.pt"
        self.normalised = load_tensor(filename)

        filename = f"{self.image_name}_confidences_patches.pt" if not self.is_filtered else f"{self.image_name}_confidences_patches_filtered.pt"
        self.confidences = load_tensor(filename)

        filename = f"{self.image_name}_which_patch.pt" if not self.is_filtered else f"{self.image_name}_which_patch_filtered.pt"
        which_patch_tensor = load_tensor(filename)
        which_patch = [(int(x), int(y)) for x, y in which_patch_tensor]
        self.which_patch = which_patch

    def save(self):
        filename = f"{self.image_name}_keypoints_normalised_patches.pt" if not self.is_filtered else f"{self.image_name}_keypoints_normalised_patches_filtered.pt"
        save_tensor(self.normalised, filename)

        filename = f"{self.image_name}_confidences_patches.pt" if not self.is_filtered else f"{self.image_name}_confidences_patches_filtered.pt"
        save_tensor(self.confidences, filename)

        filename = f"{self.image_name}_which_patch.pt" if not self.is_filtered else f"{self.image_name}_which_patch_filtered.pt"
        which_patch = torch.tensor(self.which_patch)
        save_tensor(which_patch, filename)


class Keypoints:
    def __init__(self, image_name, is_filtered=False):
        self.is_filtered = is_filtered

        self.image_path: str = f"{config.paths[config.task.name].images_dir}/{image_name}"
        self.image: Image.Image = self._init_image()
        image_name, _ = os.path.splitext(image_name)
        self.image_name: str = image_name

        patch_images, patches_shape = self._init_grid_patches()
        self.patch_images: Dict[Tuple[int, int], Image.Image] = patch_images
        self.patches_shape: Tuple[int, int] = patches_shape

        self.image_keypoints = ImageKeypoints(image_name, is_filtered=False)
        self.patches_keypoints = PatchesKeypoints(image_name, is_filtered=False)

        self.image_keypoints_filtered = ImageKeypoints(image_name, is_filtered=True)
        self.patches_keypoints_filtered = PatchesKeypoints(image_name, is_filtered=True)

    def _init_image(self):
        assert os.path.exists(self.image_path)

        image = Image.open(self.image_path)
        x, y = config.image.image_shape
        image = image.resize((x, y))
        image = image.convert('RGB')

        return image

    def _init_grid_patches(self):
        assert self.image is not None

        # image_np.shape (896, 896, 3) example
        image_np = np.array(self.image)
        # view.shape (7, 7, 1, 128, 128, 3)
        x, y = config.image.patch_shape
        view = view_as_blocks(image_np, (x, y, 3))
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

    @staticmethod
    def load_from_name(image_name, is_filtered=False):
        kd = Keypoints(image_name, is_filtered)
        kd.load()
        return kd

    """
    Getters
    """

    @staticmethod
    def _get_unique_coords(keypoints1: List[cv2.KeyPoint], keypoints2: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        coords = keypoints1 + keypoints2
        unique_coords = []
        seen = set()  # To store the unique (x, y) tuples

        for kp in coords:
            if kp.pt not in seen:
                seen.add(kp.pt)
                unique_coords.append(kp)

        return unique_coords

    def get_all_coords(self):
        assert self.image_keypoints
        x: List[cv2.KeyPoint] = self.image_keypoints.as_image_coords()
        assert self.patches_keypoints
        y: List[cv2.KeyPoint] = self.patches_keypoints.as_image_coords()

        return self._get_unique_coords(x, y)

    def get_all_filtered_coords(self):
        assert self.image_keypoints_filtered
        x: List[cv2.KeyPoint] = self.image_keypoints_filtered.as_image_coords()
        assert self.patches_keypoints_filtered
        y: List[cv2.KeyPoint] = self.patches_keypoints_filtered.as_image_coords()

        return self._get_unique_coords(x, y)

    """
    Load & Save
    """

    def load(self):
        self.image_keypoints.load()
        self.patches_keypoints.load()

        if self.is_filtered:
            self.image_keypoints_filtered.load()
            self.patches_keypoints_filtered.load()

    def save(self):
        self.image_keypoints.save()
        self.patches_keypoints.save()

        if self.is_filtered:
            self.image_keypoints_filtered.save()
            self.patches_keypoints_filtered.save()


"""
MATCHES
"""


class Matches:
    def __init__(self, a: Keypoints, b: Keypoints):
        self.a = a
        self.b = b

        self.warp: Optional[torch.tensor] = None
        self.pixel_coords: Optional[torch.tensor] = None
        self.certainty: Optional[torch.tensor] = None

        self.left_coords: Optional[List[cv2.KeyPoint]] = None
        self.right_coords: Optional[List[cv2.KeyPoint]] = None

    @staticmethod
    def load_from_names(name_a, name_b, load_coords=False):
        a = Keypoints.load_from_name(name_a)
        b = Keypoints.load_from_name(name_b)

        pair = Matches(a, b)
        pair.load()

        if load_coords:
            pair.load_coords()

        return pair

    """
    Utils
    """

    # noinspection PyArgumentList
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
        filename = f"{self.a.image_name}_{self.b.image_name}_warp.pt"
        warp = load_tensor(filename)
        self.set_warp(warp)

        filename = f"{self.a.image_name}_{self.b.image_name}_certainty.pt"
        self.certainty = load_tensor(filename)

    def save(self):
        filename = f"{self.a.image_name}_{self.b.image_name}_warp.pt"
        save_tensor(self.warp, filename)

        filename = f"{self.a.image_name}_{self.b.image_name}_certainty.pt"
        save_tensor(self.certainty, filename)

    def load_coords(self):
        filename = f"{self.a.image_name}_{self.b.image_name}_matches.pt"
        matches = load_tensor(filename)

        self.left_coords = [
            cv2.KeyPoint(int(x), int(y), 1.)
            for x, y in matches[:, :2]
        ]

        self.right_coords = [
            cv2.KeyPoint(int(x), int(y), 1.)
            for x, y in matches[:, 2:]
        ]

    def save_coords(self):
        assert self.left_coords is not None and self.right_coords is not None

        filename = f"{self.a.image_name}_{self.b.image_name}_matches.pt"

        left_coords = torch.tensor([kp.pt for kp in self.left_coords])
        right_coords = torch.tensor([kp.pt for kp in self.right_coords])

        matches = torch.cat([left_coords, right_coords], dim=1)
        save_tensor(matches, filename)
