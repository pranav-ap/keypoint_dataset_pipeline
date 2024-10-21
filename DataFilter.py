from config import config
from utils import logger
from typing import Optional
import cv2
import torch
from ImageData import KeypointsData, MatchesData
from rich.progress import Progress


class DataFilter:
    def __init__(self):
        pass

    @staticmethod
    def _good_image_level_keypoints(kd: KeypointsData):
        confidence_threshold: int = config.dedode.filter.image_confidence_threshold
        mask: torch.tensor = kd.confidences >= confidence_threshold

        confident_image_keypoints = kd.keypoints[mask]
        confident_image_confidences = kd.confidences[mask]

        logger.debug(f'kd.confidences.shape {kd.confidences.shape}')
        logger.debug(f'confident_image_confidences.shape {confident_image_confidences.shape}')

        sorted_confidence_indices = torch.argsort(confident_image_confidences, descending=True)

        keep_top_k: int = config.dedode.filter.keep_top_k
        keep_top_k = min(keep_top_k, len(sorted_confidence_indices))

        top_keypoints = confident_image_keypoints[sorted_confidence_indices][:keep_top_k]

        return top_keypoints

    @staticmethod
    def _good_patches_level_keypoints(kd: KeypointsData):
        confidence_threshold: int = config.dedode.filter.patches_confidence_threshold
        mask: torch.tensor = kd.confidences_patches >= confidence_threshold

        if any(mask):
            confident_patches_keypoints = kd.keypoints_patches[mask]

            logger.debug(f'kd.keypoints_patches.shape {kd.keypoints_patches.shape}')
            logger.debug(f'confident_patches_keypoints.shape {confident_patches_keypoints.shape}')

            return confident_patches_keypoints

        return kd.keypoints_patches

    def _good_keypoints(self, kd: KeypointsData):
        keypoints_image_level = self._good_image_level_keypoints(kd)
        logger.info(f'keypoints_image_level.shape : {keypoints_image_level.shape}')
        keypoints_patches_level = self._good_patches_level_keypoints(kd)
        logger.info(f'keypoints_patches_level.shape : {keypoints_patches_level.shape}')

        keypoints = torch.cat([keypoints_image_level, keypoints_patches_level], dim=0)

        keypoints_coords = [
            cv2.KeyPoint(
                int((x.item() + 1) * (kd.image.width / 2)),
                int((y.item() + 1) * (kd.image.height / 2)),
                1
            )
            for x, y in keypoints
        ]

        logger.info(f'Good Keypoints : {len(keypoints_coords)}')

        return keypoints_coords

    @staticmethod
    def _good_matches(pair: MatchesData, reference_keypoints_coords):
        confidence_threshold = config.roma.filter.confidence_threshold

        left_matches_coords, right_matches_coords = pair.get_good_matches(
            reference_keypoints_coords,
            confidence_threshold
        )

        assert len(left_matches_coords) == len(right_matches_coords)

        logger.info(f'Good Matches : {len(right_matches_coords)}')

        pair.left_matches_coords_filtered = left_matches_coords
        pair.right_matches_coords_filtered = right_matches_coords


    def extract_good_matches(self, image_names):
        with Progress() as progress:
            task = progress.add_task(
                "[red]Extracting matches...",
                total=len(image_names) - 1
            )

            a: Optional[KeypointsData] = None
            top_keypoints = None

            for name_a, name_b in zip(image_names, image_names[1:]):
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
                pair.save_filtered_matches()

                # Update for next iteration
                top_keypoints = self._good_keypoints(b)
                a = b

                progress.advance(task)

            progress.stop()
