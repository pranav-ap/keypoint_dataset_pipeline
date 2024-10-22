from config import config
from utils import logger
from typing import Optional
import torch
from ImageData import Keypoints, Matches
from rich.progress import Progress


class DataFilter:
    @staticmethod
    def _filter_image_level_keypoints(kd: Keypoints):
        keypoints = kd.image_keypoints.normalised
        confidences = kd.image_keypoints.confidences

        if len(keypoints) == 0:
            return torch.empty(0)

        count = min(config.dedode.filter.image_level_max_count, len(keypoints))
        selected_indices = torch.multinomial(confidences, count, replacement=False)

        top_keypoints = keypoints[selected_indices]
        top_confidences = keypoints[selected_indices]

        kd.image_keypoints_filtered.normalised = top_keypoints
        kd.image_keypoints_filtered.confidences = top_confidences

    @staticmethod
    def _filter_patches_level_keypoints(kd: Keypoints):
        top_keypoints = kd.patches_keypoints.normalised
        top_confidences = kd.patches_keypoints.confidences

        if len(top_keypoints) == 0:
            return torch.empty(0)

        kd.patches_keypoints_filtered.normalised = top_keypoints
        kd.patches_keypoints_filtered.confidences = top_confidences

    def _filter_keypoints(self, kd: Keypoints):
        self._filter_image_level_keypoints(kd)
        self._filter_patches_level_keypoints(kd)
        kd.is_filtered = True
        kd.save()

        coords = kd.get_all_filtered_coords()
        logger.info(f'Good Keypoints : {len(coords)}')

        return coords

    @staticmethod
    def _filter_matches(pair: Matches, reference_keypoints_coords):
        threshold = config.roma.filter.confidence_threshold

        left_coords, right_coords = pair.get_good_matches(
            reference_keypoints_coords,
            threshold
        )

        assert len(left_coords) == len(right_coords)
        logger.info(f'Good Matches : {len(right_coords)}')

        pair.left_coords = left_coords
        pair.right_coords = right_coords

    def extract_good_matches(self, image_names):
        with Progress() as progress:
            task = progress.add_task(
                "[red]Extracting matches...",
                total=len(image_names) - 1
            )

            a: Optional[Keypoints] = None
            top_keypoints = None

            for index, (name_a, name_b) in enumerate(zip(image_names, image_names[1:])):
                if a is None:
                    a = Keypoints(name_a)
                    a.load()

                    top_keypoints = self._filter_keypoints(a)

                b = Keypoints(name_b)
                b.load()

                # Load matches data between a and b
                pair = Matches(a, b)
                pair.load()

                # Filter good matches based on top keypoints
                self._filter_matches(pair, top_keypoints)
                pair.save_coords()

                # Update for next iteration
                if index < len(image_names) - 2:
                    top_keypoints = self._filter_keypoints(b)

                a = b
                progress.advance(task)

            progress.stop()
