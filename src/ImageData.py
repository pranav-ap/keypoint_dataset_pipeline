import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from config import config
from utils import get_best_device
from .DataStore import DataStore

"""
KEYPOINTS
"""


class _Keypoints(ABC):
    device = get_best_device()

    def __init__(self, image_name, data_store, is_filtered):
        self.image_name: str = image_name
        self.data_store = data_store
        self.is_filtered: bool = is_filtered

        self.normalised: torch.Tensor = torch.empty(0, device=self.device)
        self.confidences: torch.Tensor = torch.empty(0, device=self.device)

    @abstractmethod
    def as_image_coords(self) -> List[cv2.KeyPoint]:
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass


class ImageKeypoints(_Keypoints):
    def __init__(self, image_name, data_store, is_filtered=False):
        super().__init__(image_name, data_store, is_filtered)

    def as_image_coords(self) -> List[cv2.KeyPoint]:
        w, h = config.image.crop_image_shape

        if self.normalised.numel() == 0:
            return []

        coords = [
            cv2.KeyPoint(
                (x.item() + 1) * (w / 2),
                (y.item() + 1) * (h / 2),
                1.
            )
            for x, y in self.normalised
        ]

        return coords

    def load(self):
        normalised = self.data_store.filter_normalised[self.image_name][()] if self.is_filtered else \
            self.data_store.detector_normalised[self.image_name][()]
        self.normalised = torch.from_numpy(normalised)

        confidences = self.data_store.filter_confidences[self.image_name][()] if self.is_filtered else \
            self.data_store.detector_confidences[self.image_name][()]
        self.confidences = torch.from_numpy(confidences)

    def save(self):
        assert self.normalised is not None
        g = self.data_store.filter_normalised if self.is_filtered else self.data_store.detector_normalised
        if self.image_name not in g:
            data = self.normalised.cpu().numpy()
            g.create_dataset(self.image_name, data=data, compression='gzip', compression_opts=9)

        assert self.confidences is not None
        g = self.data_store.filter_confidences if self.is_filtered else self.data_store.detector_confidences
        if self.image_name not in g:
            data = self.confidences.cpu().numpy()
            g.create_dataset(self.image_name, data=data, compression='gzip', compression_opts=9)


class Keypoints:
    def __init__(self, image_name, data_store: DataStore, is_filtered=False, must_crop=True):
        self.is_filtered = is_filtered
        self.must_crop = must_crop

        self.image_name: str = str(image_name).strip()
        self.image_path: str = f"{config.paths[config.task.name].images}/{image_name}"

        if config.task.name == 'basalt':
            self.image_path = f"{self.image_path}.png"

        self.original_image, self.image = self._init_image()

        self.image_keypoints = ImageKeypoints(self.image_name, data_store, is_filtered=False)
        self.image_keypoints_filtered = ImageKeypoints(self.image_name, data_store, is_filtered=True)

    @staticmethod
    def crop_from_center(image, crop_width, crop_height):
        img_width, img_height = image.size

        center_x, center_y = img_width // 2, img_height // 2

        left = center_x - crop_width // 2
        top = center_y - crop_height // 2
        right = center_x + crop_width // 2
        bottom = center_y + crop_height // 2

        cropped_image = image.crop((left, top, right, bottom))

        return cropped_image

    def _init_image(self) -> Tuple[Image.Image, Image.Image]:
        assert os.path.exists(self.image_path), self.image_path
        image = Image.open(self.image_path)
        original_image = image.copy()

        if self.must_crop:
            w, h = config.image.crop_image_shape
            image = self.crop_from_center(image, w, h)

        image = image.convert('RGB')

        return original_image, image

    @staticmethod
    def load_from_name(image_name, data_store, is_filtered=False, must_crop=True):
        kd = Keypoints(image_name, data_store, is_filtered=is_filtered, must_crop=must_crop)
        kd.load()
        return kd

    """
    Getters
    """

    @staticmethod
    def _get_unique_coords(keypoints1: List[cv2.KeyPoint], keypoints2: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        coords = keypoints1 + keypoints2

        unique_coords = []
        seen = set()

        for kp in coords:
            if kp.pt not in seen:
                seen.add(kp.pt)
                unique_coords.append(kp)

        return unique_coords

    def get_all_coords(self):
        assert self.image_keypoints
        x: List[cv2.KeyPoint] = self.image_keypoints.as_image_coords()
        return x

    def get_all_filtered_coords(self):
        assert self.image_keypoints_filtered
        x: List[cv2.KeyPoint] = self.image_keypoints_filtered.as_image_coords()
        return x

    """
    Load & Save
    """

    def load(self):
        self.image_keypoints.load()

        if self.is_filtered:
            self.image_keypoints_filtered.load()

    def save(self):
        self.image_keypoints.save()

        if self.is_filtered:
            self.image_keypoints_filtered.save()


"""
MATCHES
"""


class Matches:
    def __init__(self, a: Keypoints, b: Keypoints, data_store, kpid=None, saves=None):
        self.a = a
        self.b = b

        self.kpid = kpid
        self.saves=saves

        self.data_store = data_store

        self.warp: Optional[torch.tensor] = None
        self.pixel_coords: Optional[torch.tensor] = None
        self.certainty: Optional[torch.tensor] = None

        self.reference_crop_coords: Optional[List[cv2.KeyPoint]] = None
        self.target_crop_coords: Optional[List[cv2.KeyPoint]] = None

        self.rotations = None

    @staticmethod
    def load_from_names(name_a, name_b, data_store, load_coords=False, must_crop=True):
        a = Keypoints.load_from_name(name_a, data_store, must_crop=must_crop)
        b = Keypoints.load_from_name(name_b, data_store, must_crop=must_crop)

        pair = Matches(a, b, data_store)
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
        
        if not config.task.only_missing:
            self.pixel_coords = self._warp_to_pixel_coords()

    def get_good_matches(self, reference_keypoints: List[cv2.KeyPoint], confidence_threshold=0.6) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint]]:
        """
        Just use for testing. Logic incorrect
        """

        target_keypoints = []
        accepted_reference_keypoints = []

        for pt in reference_keypoints:
            query_x, query_y = pt.pt
            rounded_query_y, rounded_query_x = round(query_y), round(query_x)

            conf = self.certainty[rounded_query_y, rounded_query_x]
            if conf <= confidence_threshold:
                continue

            closest_ref_x, closest_ref_y, closest_tar_x, closest_tar_y = self.pixel_coords[rounded_query_y, rounded_query_x]
            closest_ref_x, closest_ref_y, closest_tar_x, closest_tar_y = closest_ref_x.item(), closest_ref_y.item(), closest_tar_x.item(), closest_tar_y.item()

            ref_y_diff = query_y - closest_ref_y
            ref_x_diff = query_x - closest_ref_x

            answer_y = closest_tar_y + ref_y_diff
            answer_x = closest_tar_x + ref_x_diff

            accepted_reference_keypoints.append(pt)
            target_keypoints.append(cv2.KeyPoint(answer_x, answer_y, 1.))

        return accepted_reference_keypoints, target_keypoints

    def get_coords_on_original_image(self) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint]]:
        original_w, original_h = config.image.original_image_shape
        crop_w, crop_h = config.image.crop_image_shape

        print(f'original_w, original_h {original_w, original_h}')
        print(f'crop_w, crop_h {crop_w, crop_h}')

        left_padding = (original_w - crop_w) / 2
        top_padding = (original_h - crop_h) / 2

        print(f'left_padding, top_padding {left_padding, top_padding}')

        reference_coords = [
            cv2.KeyPoint(kp.pt[0] + left_padding, kp.pt[1] + top_padding, 1.)
            for kp in self.reference_crop_coords
        ]

        target_coords = [
            cv2.KeyPoint(kp.pt[0] + left_padding, kp.pt[1] + top_padding, 1.)
            for kp in self.target_crop_coords
        ]

        return reference_coords, target_coords

    """
    Load & Save
    """

    def load(self):
        pair_name = f"{self.a.image_name}_{self.b.image_name}"

        if self.kpid is not None:
            pair_name = f"{self.a.image_name}_{self.b.image_name}_{self.kpid}"
            self.saves = self.data_store.matcher_saves[pair_name][()]

        warp = self.data_store.matcher_warp[pair_name][()]
        warp = torch.from_numpy(warp)
        self.set_warp(warp)

        self.certainty = self.data_store.matcher_certainty[pair_name][()]

    def save(self):
        pair_name = f"{self.a.image_name}_{self.b.image_name}"

        if self.kpid is not None:
            pair_name = f"{self.a.image_name}_{self.b.image_name}_{self.kpid}"

            assert self.saves is not None
            g = self.data_store.matcher_saves
            if pair_name not in g:
                data = self.saves
                g.create_dataset(pair_name, data=data, compression='gzip', compression_opts=9)

        assert self.warp is not None
        g = self.data_store.matcher_warp
        if pair_name not in g:
            data = self.warp.cpu().numpy()
            g.create_dataset(pair_name, data=data, compression='gzip', compression_opts=9)

        assert self.certainty is not None
        g = self.data_store.matcher_certainty
        if pair_name not in g:
            data = self.certainty.cpu().numpy()
            g.create_dataset(pair_name, data=data, compression='gzip', compression_opts=9)

    def load_coords(self):
        pair_name = f"{self.a.image_name}_{self.b.image_name}"

        reference_crop_coords = self.data_store.crop_reference_coords[pair_name][()]
        self.reference_crop_coords = [
            cv2.KeyPoint(x, y, 1.)
            for x, y in reference_crop_coords
        ]

        target_crop_coords = self.data_store.crop_target_coords[pair_name][()]
        self.target_crop_coords = [
            cv2.KeyPoint(x, y, 1.)
            for x, y in target_crop_coords
        ]

    def save_coords(self):
        assert self.reference_crop_coords is not None
        assert self.target_crop_coords is not None

        reference_crop_coords = np.array([(kp.pt[0], kp.pt[1]) for kp in self.reference_crop_coords])
        target_crop_coords = np.array([(kp.pt[0], kp.pt[1]) for kp in self.target_crop_coords])

        pair_name = f"{self.a.image_name}_{self.b.image_name}"

        g = self.data_store.crop_reference_coords
        if pair_name not in g:
            g.create_dataset(pair_name, data=reference_crop_coords, compression='gzip', compression_opts=9)

        g = self.data_store.crop_target_coords
        if pair_name not in g:
            g.create_dataset(pair_name, data=target_crop_coords, compression='gzip', compression_opts=9)

    def save_rotations(self):
        assert self.rotations is not None
        rotations = np.array(self.rotations)

        pair_name = f"{self.a.image_name}_{self.b.image_name}"

        g = self.data_store.rotations
        if pair_name not in g:
            g.create_dataset(pair_name, data=rotations, compression='gzip', compression_opts=9)
