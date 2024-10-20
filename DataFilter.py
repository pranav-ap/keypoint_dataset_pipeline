import torch
from config import config
from ImageData import KeypointsData, MatchesData
from rich.progress import Progress


class DataFilter:
    def __init__(self):
        pass

    @staticmethod
    def _good_image_level_keypoints(kd: KeypointsData):
        confidence_threshold = config.dedode.filter.image_confidence_threshold
        mask: torch.tensor = kd.confidences >= confidence_threshold

        confident_image_keypoints = kd.keypoints[mask]
        confident_image_confidences = kd.confidences[mask]

        if config.dedode.filter.top_k_strategy == 'sorting':
            sorted_confidence_indices = torch.argsort(confident_image_confidences, descending=True)

            keep_top_k = config.dedode.filter.keep_top_k
            keep_top_k = min(keep_top_k, len(sorted_confidence_indices))

            top_keypoints = confident_image_keypoints[sorted_confidence_indices]
            top_keypoints = top_keypoints[:keep_top_k]

            return top_keypoints

        # Ensure confidences sum to 1 for weighted sampling
        confidence_weights = confident_image_confidences / confident_image_confidences.sum()

        keep_top_k = config.dedode.filter.keep_top_k
        keep_top_k = min(keep_top_k, len(confident_image_keypoints))

        # Random sampling based on confidence weights
        selected_indices = torch.multinomial(confidence_weights, keep_top_k, replacement=False)
        top_keypoints = confident_image_keypoints[selected_indices]

        return top_keypoints

    @staticmethod
    def _good_patches_level_keypoints(kd: KeypointsData):
        confidence_threshold = config.dedode.filter.patches_confidence_threshold
        mask: torch.tensor = kd.confidences_patches >= confidence_threshold

        confident_patches_keypoints = kd.keypoints_patches[mask]
        confident_patches_confidences = kd.confidences_patches[mask]

        sorted_confidence_indices = torch.argsort(confident_patches_confidences, descending=True)
        top_keypoints = confident_patches_keypoints[sorted_confidence_indices]

        return top_keypoints

    def _good_keypoints(self, kd: KeypointsData):
        keypoints_image_level = self._good_image_level_keypoints(kd)
        keypoints_patches_level = self._good_patches_level_keypoints(kd)
        keypoints = torch.cat([keypoints_image_level, keypoints_patches_level], dim=1)

        return keypoints

    @staticmethod
    def _good_matches(pair: MatchesData, reference_keypoints_coords):
        confidence_threshold = config.roma.filter.confidence_threshold

        left_matches_coords, right_matches_coords = pair.get_good_matches(
            reference_keypoints_coords,
            confidence_threshold
        )

        pair.left_matches_coords_filtered = left_matches_coords
        pair.right_matches_coords_filtered = right_matches_coords

    def extract_good_matches(self, image_names):
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Extracting matches...",
                total=len(image_names)
            )

            name_a = image_names[0]
            a = KeypointsData(name_a)
            a.load()

            top_keypoints = self._good_keypoints(a)

            for name_b in image_names[1:]:
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
