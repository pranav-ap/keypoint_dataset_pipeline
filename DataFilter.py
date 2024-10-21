from config import config
from utils import logger
from typing import Optional
import cv2
import torch
from ImageData import KeypointsData, MatchesData
from rich.progress import Progress


class DataFilter:
    @staticmethod
    def _good_image_level_keypoints(kd: KeypointsData):
        threshold: int = config.dedode.filter.image_relative_confidence_threshold
        mask: torch.tensor = kd.image_keypoints.confidences >= threshold

        keypoints = kd.image_keypoints.normalised[mask]
        confidences = kd.image_keypoints.confidences[mask]

        if len(keypoints) == 0:
            return torch.empty(0)

        count = min(config.dedode.filter.max_count, len(keypoints))
        selected_indices = torch.multinomial(confidences, count, replacement=False)
        top_keypoints = keypoints[selected_indices]

        return top_keypoints

    @staticmethod
    def _good_patches_level_keypoints(kd: KeypointsData):
        threshold: int = config.dedode.filter.patches_relative_confidence_threshold
        mask: torch.tensor = kd.patches_keypoints.confidences >= threshold

        top_keypoints = kd.patches_keypoints.normalised[mask]

        if len(top_keypoints) == 0:
            return torch.empty(0)

        return top_keypoints

    def _good_keypoints(self, kd: KeypointsData):
        keypoints_image_level = self._good_image_level_keypoints(kd)
        keypoints_patches_level = self._good_patches_level_keypoints(kd)

        keypoints = torch.cat([keypoints_image_level, keypoints_patches_level], dim=0)

        coords = [
            cv2.KeyPoint(
                int((x.item() + 1) * (kd.image.width / 2)),
                int((y.item() + 1) * (kd.image.height / 2)),
                1
            )
            for x, y in keypoints
        ]

        logger.info(f'Good Keypoints : {len(coords)}')

        return coords

    @staticmethod
    def _good_matches(pair: MatchesData, reference_keypoints_coords):
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

            a: Optional[KeypointsData] = None
            top_keypoints = None

            for index, (name_a, name_b) in enumerate(zip(image_names, image_names[1:])):
                if a is None:
                    a = KeypointsData(name_a)
                    a.load()

                    top_keypoints = self._good_keypoints(a)

                b = KeypointsData(name_b)
                b.load()

                # Load matches data between a and b
                pair = MatchesData(a, b)
                pair.load()

                # Filter good matches based on top keypoints
                self._good_matches(pair, top_keypoints)
                pair.save_coords()

                # Update for next iteration
                if index < len(image_names) - 2:
                    top_keypoints = self._good_keypoints(b)

                a = b
                progress.advance(task)

            progress.stop()
