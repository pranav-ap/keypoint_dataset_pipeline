from typing import Optional

import torch
from tqdm import tqdm

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

    @staticmethod
    def _filter_patch_level_keypoints(kd: Keypoints):
        keypoints = kd.patches_keypoints.normalised
        confidences = kd.patches_keypoints.confidences
        which_patch = kd.patches_keypoints.which_patch

        count = min(config.dedode.filter.patches_sample_count, len(keypoints))

        if keypoints is None or count == 0:
            return

        selected_indices = torch.multinomial(confidences, count, replacement=False)

        top_keypoints = keypoints[selected_indices]
        top_confidences = confidences[selected_indices]
        top_which_patch = [which_patch[i] for i in selected_indices]

        kd.patches_keypoints_filtered.normalised = top_keypoints
        kd.patches_keypoints_filtered.confidences = top_confidences
        kd.patches_keypoints_filtered.which_patch = top_which_patch

    def _filter_keypoints(self, kd: Keypoints):
        self._filter_image_level_keypoints(kd)
        self._filter_patch_level_keypoints(kd)

        kd.is_filtered = True
        kd.save()

        coords = kd.get_all_filtered_coords()
        # logger.info(f'Good Keypoints : {len(coords)}')

        return coords

    @staticmethod
    def _filter_matches(pair: Matches, reference_keypoints_coords):
        threshold = config.roma.filter.confidence_threshold

        small_left_coords, small_right_coords = pair.get_good_matches(
            reference_keypoints_coords,
            threshold
        )

        assert len(small_left_coords) == len(small_right_coords)
        # logger.info(f'Good Matches : {len(right_coords)}')

        pair.small_left_coords = small_left_coords
        pair.small_right_coords = small_right_coords

    def extract_good_matches(self, image_names):
        a: Optional[Keypoints] = None
        top_keypoints = None

        for index, (name_a, name_b) in tqdm(enumerate(zip(image_names, image_names[1:])), desc="Extracting matches",
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

            a = b
