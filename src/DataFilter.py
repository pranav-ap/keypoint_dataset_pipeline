from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from .rotate import solve_patch_rotation
from config import config
from .ImageData import Keypoints, Matches


class DataFilter:
    def __init__(self, data_store):
        self.data_store = data_store

    @staticmethod
    def _filter_image_level_keypoints(kd: Keypoints):
        keypoints = kd.image_keypoints.normalised
        confidences = kd.image_keypoints.confidences

        count = min(config.dedode.filter.images_sample_count, len(keypoints))

        if keypoints is None or count == 0:
            return

        selected_indices = torch.multinomial(confidences, count, replacement=False)

        top_keypoints = keypoints[selected_indices]
        top_confidences = confidences[selected_indices]

        kd.image_keypoints_filtered.normalised = top_keypoints
        kd.image_keypoints_filtered.confidences = top_confidences

    def _filter_keypoints(self, kd: Keypoints):
        self._filter_image_level_keypoints(kd)

        kd.is_filtered = True
        kd.save()

        coords = kd.get_all_filtered_coords()
        # logger.info(f'Good Keypoints : {len(coords)}')

        return coords

    @staticmethod
    def _filter_matches(pair: Matches, reference_keypoints_coords):
        threshold = config.roma.filter.confidence_threshold

        reference_crop_coords, target_crop_coords = pair.get_good_matches(
            reference_keypoints_coords,
            threshold
        )

        assert len(reference_crop_coords) == len(target_crop_coords)
        # logger.info(f'Good Matches : {len(right_coords)}')

        pair.reference_crop_coords = reference_crop_coords
        pair.target_crop_coords = target_crop_coords

    def extract_good_matches(self, image_names):
        a: Optional[Keypoints] = None
        top_keypoints = None

        for index, (name_a, name_b) in tqdm(enumerate(zip(image_names, image_names[1:])),
                                            desc="Extracting matches",
                                            ncols=100, total=len(image_names) - 1):
            if a is None:
                a = Keypoints.load_from_name(name_a, self.data_store)
                top_keypoints = self._filter_keypoints(a)

            b = Keypoints.load_from_name(name_b, self.data_store)

            # Load matches data between a and b
            pair = Matches(a, b, self.data_store)
            pair.load()

            # Filter good matches based on top keypoints
            self._filter_matches(pair, top_keypoints)
            pair.save_coords()

            # Update for next iteration
            top_keypoints = self._filter_keypoints(b)

            # Extract rotations

            reference_crop_coords, target_crop_coords = pair.get_coords_on_original_image()
            pair.rotations = []

            for i in range(len(reference_crop_coords)):
                angle = solve_patch_rotation(
                    a.original_image, b.original_image,
                    np.array([reference_crop_coords[i].pt[0], reference_crop_coords[i].pt[1]]),
                    np.array([target_crop_coords[i].pt[0], target_crop_coords[i].pt[1]]),
                )

                pair.rotations.append(angle)

            pair.save_rotations()

            a = b
