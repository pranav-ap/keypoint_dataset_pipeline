from config import config
from utils import logger, get_best_device, make_clear_directory
import cv2
import os
import torch
import numpy as np
from PIL import Image
from skimage.util import view_as_blocks
from typing import List, Optional, Tuple, Dict
from abc import ABC, abstractmethod
import h5py


class DataStore:
    def __init__(self, mode='a'):
        assert mode == 'r' or mode =='a'
        
        filename_inter = 'inter.hdf5'
        filepath_inter = f'{config.paths[config.task.name].output}/{filename_inter}'
        self._file_inter = h5py.File(filepath_inter, mode)

        filename_results = 'results.hdf5'
        filepath_results = f'{config.paths[config.task.name].output}/{filename_results}'
        self._file_results = h5py.File(filepath_results, mode)

        if mode == 'r':
            self._init_groups_read_mode()
        else:
            self._init_groups_append_mode()

    def _init_groups_append_mode(self):
        # Create groups in the interaction file
        self._detector = self._file_inter.create_group('detector')
        self._matcher = self._file_inter.create_group('matcher')
        self._filter = self._file_inter.create_group('filter')

        # Create groups in the results file
        self._results_matches = self._file_results.create_group('matches')

        # Setup 'detector' subgroups
        self.detector_image_level_normalised = self._detector.create_group('image_level/normalised')
        self.detector_image_level_confidences = self._detector.create_group('image_level/confidences')
        self.detector_patch_level_normalised = self._detector.create_group('patch_level/normalised')
        self.detector_patch_level_confidences = self._detector.create_group('patch_level/confidences')
        self.detector_patch_level_which_patch = self._detector.create_group('patch_level/which_patch')

        # Setup 'matcher' subgroups
        self.matcher_warp = self._matcher.create_group('warp')
        self.matcher_certainty = self._matcher.create_group('certainty')

        # Setup 'filter' subgroups
        self.filter_image_level_normalised = self._filter.create_group('image_level/normalised')
        self.filter_image_level_confidences = self._filter.create_group('image_level/confidences')
        self.filter_patch_level_normalised = self._filter.create_group('patch_level/normalised')
        self.filter_patch_level_confidences = self._filter.create_group('patch_level/confidences')
        self.filter_patch_level_which_patch = self._filter.create_group('patch_level/which_patch')

        # Setup results subgroups
        self.results_reference_coords = self._results_matches.create_group('reference_coords')
        self.results_target_coords = self._results_matches.create_group('target_coords')
    
    def _init_groups_read_mode(self):
        # Create groups in the interaction file
        self._detector = self._file_inter['detector']
        self._matcher = self._file_inter['matcher']
        self._filter = self._file_inter['filter']

        # Create groups in the results file
        self._results_matches = self._file_results['matches']

        # Setup 'detector' subgroups
        self.detector_image_level_normalised = self._detector['image_level/normalised']
        self.detector_image_level_confidences = self._detector['image_level/confidences']
        self.detector_patch_level_normalised = self._detector['patch_level/normalised']
        self.detector_patch_level_confidences = self._detector['patch_level/confidences']
        self.detector_patch_level_which_patch = self._detector['patch_level/which_patch']

        # Setup 'matcher' subgroups
        self.matcher_warp = self._matcher['warp']
        self.matcher_certainty = self._matcher['certainty']

        # Setup 'filter' subgroups
        self.filter_image_level_normalised = self._filter['image_level/normalised']
        self.filter_image_level_confidences = self._filter['image_level/confidences']
        self.filter_patch_level_normalised = self._filter['patch_level/normalised']
        self.filter_patch_level_confidences = self._filter['patch_level/confidences']
        self.filter_patch_level_which_patch = self._filter['patch_level/which_patch']

        # Setup results subgroups
        self.results_reference_coords = self._results_matches['reference_coords']
        self.results_target_coords = self._results_matches['target_coords']

    def close(self):
        self._file_inter.close()
        self._file_results.close()


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

    @abstractmethod
    def as_image_coords(self) -> List[cv2.KeyPoint]:
        pass

    @abstractmethod
    def load(self, data_store: DataStore):
        pass

    @abstractmethod
    def save(self, data_store: DataStore):
        pass


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

    def load(self, data_store):
        normalised = data_store.filter_image_level_normalised[self.image_name][()] if self.is_filtered else data_store.detector_image_level_normalised[self.image_name][()]
        self.normalised = torch.from_numpy(normalised)

        confidences = data_store.filter_image_level_confidences[self.image_name][()] if self.is_filtered else data_store.detector_image_level_confidences[self.image_name][()]
        self.confidences = torch.from_numpy(confidences)

    def save(self, data_store):
        assert self.normalised is not None
        g = data_store.filter_image_level_normalised if self.is_filtered else data_store.detector_image_level_normalised
        if not self.image_name in g:
            data = self.normalised.cpu().numpy()
            g.create_dataset(self.image_name, data=data)

        assert self.confidences is not None
        g = data_store.filter_image_level_confidences if self.is_filtered else data_store.detector_image_level_confidences
        if not self.image_name in g:
            data = self.confidences.cpu().numpy()
            g.create_dataset(self.image_name, data=data)


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

    def load(self, data_store):
        normalised = data_store.filter_patch_level_normalised[self.image_name][()] if self.is_filtered else data_store.detector_patch_level_normalised[self.image_name][()]
        self.normalised = torch.from_numpy(normalised)

        confidences = data_store.filter_patch_level_confidences[self.image_name][()] if self.is_filtered else data_store.detector_patch_level_confidences[self.image_name][()]
        self.confidences = torch.from_numpy(confidences)

        which_patch = data_store.filter_patch_level_which_patch[self.image_name][()] if self.is_filtered else data_store.detector_patch_level_which_patch[self.image_name][()]
        self.which_patch = [(int(x), int(y)) for x, y in which_patch]

    def save(self, data_store):
        assert self.normalised is not None
        g = data_store.filter_patch_level_normalised if self.is_filtered else data_store.detector_patch_level_normalised
        if not self.image_name in g:
            data = self.normalised.cpu().numpy()
            g.create_dataset(self.image_name, data=data)

        assert self.confidences is not None
        g = data_store.filter_patch_level_confidences if self.is_filtered else data_store.detector_patch_level_confidences
        if not self.image_name in g:
            data = self.confidences.cpu().numpy()
            g.create_dataset(self.image_name, data=data)

        assert self.which_patch is not None
        g = data_store.filter_patch_level_which_patch if self.is_filtered else data_store.detector_patch_level_which_patch
        if not self.image_name in g:
            data = self.which_patch
            g.create_dataset(self.image_name, data=data)


class Keypoints:
    def __init__(self, image_name, is_filtered=False, must_resize=True, no_patches=False):
        self.is_filtered = is_filtered
        self.must_resize = must_resize

        self.image_name: str = str(image_name).strip()
        self.image_path: str = f"{config.paths[config.task.name].images}/{image_name}"

        if config.task.name == 'basalt':
            self.image_path = f"{self.image_path}.png"

        self.image: Image.Image = self._init_image()

        self.image_keypoints = ImageKeypoints(self.image_name, is_filtered=False)
        self.image_keypoints_filtered = ImageKeypoints(self.image_name, is_filtered=True)

        self.patch_images = None
        self.patches_shape = None
        self.patches_keypoints = None
        self.patches_keypoints_filtered = None

        if not no_patches:
            patch_images, patches_shape = self._init_grid_patches()
            self.patch_images: Dict[Tuple[int, int], Image.Image] = patch_images
            self.patches_shape: Tuple[int, int] = patches_shape
    
            self.patches_keypoints = PatchesKeypoints(self.image_name, is_filtered=False)
            self.patches_keypoints_filtered = PatchesKeypoints(self.image_name, is_filtered=True)

    def _init_image(self):
        assert os.path.exists(self.image_path)

        image = Image.open(self.image_path)
        x, y = config.image.image_shape

        if self.must_resize:
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
    def load_from_name(image_name, data_store, is_filtered=False, no_patches=False, must_resize=True):
        kd = Keypoints(image_name, is_filtered=is_filtered, no_patches=no_patches, must_resize=must_resize)
        kd.load(data_store)
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
        assert self.patches_keypoints
        y: List[cv2.KeyPoint] = self.patches_keypoints.as_image_coords()

        return self._get_unique_coords(x, y)

    def get_all_filtered_coords(self):
        assert self.image_keypoints_filtered
        x: List[cv2.KeyPoint] = self.image_keypoints_filtered.as_image_coords()
        assert self.image_keypoints_filtered
        y: List[cv2.KeyPoint] = self.patches_keypoints_filtered.as_image_coords()

        return self._get_unique_coords(x, y)

    """
    Load & Save
    """

    def load(self, data_store):
        self.image_keypoints.load(data_store)

        if self.patches_keypoints:
            self.patches_keypoints.load(data_store)

        if self.is_filtered:
            self.image_keypoints_filtered.load(data_store)
            
            if self.patches_keypoints_filtered:
                self.patches_keypoints_filtered.load(data_store)

    def save(self, data_store):
        self.image_keypoints.save(data_store)

        if self.patches_keypoints:
            self.patches_keypoints.save(data_store)

        if self.is_filtered:
            self.image_keypoints_filtered.save(data_store)

            if self.patches_keypoints_filtered:
                self.patches_keypoints_filtered.save(data_store)


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
    def load_from_names(name_a, name_b, data_store, load_coords=False, no_patches=False, must_resize=True):
        a = Keypoints.load_from_name(name_a, data_store, no_patches=no_patches, must_resize=must_resize)
        b = Keypoints.load_from_name(name_b, data_store, no_patches=no_patches, must_resize=must_resize)

        pair = Matches(a, b)
        pair.load(data_store)

        if load_coords:
            pair.load_coords(data_store)

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

    def load(self, data_store: DataStore):
        pair_name = f"{self.a.image_name}_{self.b.image_name}"

        warp = data_store.matcher_warp[pair_name][()]
        warp = torch.from_numpy(warp)
        self.set_warp(warp)

        self.certainty = data_store.matcher_certainty[pair_name][()]

    def save(self, data_store: DataStore):
        pair_name = f"{self.a.image_name}_{self.b.image_name}"

        assert self.warp is not None
        g = data_store.matcher_warp
        if not pair_name in g:
            data = self.warp.cpu().numpy()
            g.create_dataset(pair_name, data=data)

        assert self.certainty is not None
        g = data_store.matcher_certainty
        data = self.certainty.cpu().numpy()
        g.create_dataset(pair_name, data=data)

    def load_coords(self, data_store: DataStore):
        pair_name = f"{self.a.image_name}_{self.b.image_name}"

        left_coords = data_store.results_reference_coords[pair_name][()]
        self.left_coords = [
            cv2.KeyPoint(int(x), int(y), 1.)
            for x, y in left_coords
        ]

        right_coords = data_store.results_target_coords[pair_name][()]
        self.right_coords = [
            cv2.KeyPoint(int(x), int(y), 1.)
            for x, y in right_coords
        ]

    def save_coords(self, data_store: DataStore):
        assert self.left_coords is not None
        assert self.right_coords is not None

        current_width, current_height = config.image.image_shape
        new_width, new_height = config.image.original_image_shape

        def resize_coordinates(x, y):
            new_x = x * (new_width / current_width)
            new_y = y * (new_height / current_height)
            kp = round(new_x), round(new_y)
            return kp

        left_coords = np.array([resize_coordinates(kp.pt[0], kp.pt[1]) for kp in self.left_coords])
        right_coords = np.array([resize_coordinates(kp.pt[0], kp.pt[1]) for kp in self.right_coords])

        pair_name = f"{self.a.image_name}_{self.b.image_name}"

        g = data_store.results_reference_coords
        if not pair_name in g:
            data = left_coords
            g.create_dataset(pair_name, data=data)

        g = data_store.results_target_coords
        if not pair_name in g:
            data = right_coords
            g.create_dataset(pair_name, data=data)
