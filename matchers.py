import cv2
from config import config
from ImageData import ImageSoloData, DSM_ImagePairData, RoMa_ImagePairData
from visualizer import Visualizer
from typing import Optional
from abc import ABC, abstractmethod


class KeypointMatcher(ABC):
    def __init__(self):
        pass


class DSMatcher(KeypointMatcher):
    def __init__(self):
        super().__init__()

        from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
        self.matcher = DualSoftMaxMatcher()

    """
    Match
    """

    def _match_pair(self, pair: DSM_ImagePairData):
        a: ImageSoloData = pair.a
        b: ImageSoloData = pair.b

        matches_A, matches_B, batch_ids = self.matcher.match(
            a.keypoints, a.descriptions,
            b.keypoints, b.descriptions,
            normalize=True,
            inv_temp=20,
            threshold=0.1
        )

        H = a.image.height
        W = a.image.width

        left_matches, right_matches = self.matcher.to_pixel_coords(
            matches_A, matches_B,
            H, W, H, W
        )

        pair.set_left_matches(left_matches)
        pair.set_right_matches(right_matches)

    def extract_matches(self, image_names):
        a: Optional[ImageSoloData] = None

        for index in range(len(image_names) - 1):
            path_a = f"{config.images_dir_path}/{image_names[index]}"
            path_b = f"{config.images_dir_path}/{image_names[index + 1]}"

            if a is None:
                a = ImageSoloData(path_a)
                a.load_keypoints()

            b = ImageSoloData(path_b)
            b.load_keypoints()

            pair = DSM_ImagePairData(a, b)
            self._match_pair(pair)
            pair.save_matches()

            a = b

    @staticmethod
    def show_matches(path_a, path_b, num_points=10):
        a = ImageSoloData(path_a)
        a.load_keypoints()

        b = ImageSoloData(path_b)
        b.load_keypoints()

        pair = DSM_ImagePairData(a, b)
        pair.load_matches()

        return Visualizer.plot_matches(pair, num_points)


class RoMaMatcher(KeypointMatcher):
    def __init__(self):
        super().__init__()

        from romatch import roma_outdoor
        self.model = roma_outdoor(
            device=config.device,
            coarse_res=560,
            upsample_res=config.IMAGE_RESIZE
        )

        self.model.symmetric = False

    def extract_warp_certainty(self, image_names):
        a: Optional[ImageSoloData] = None

        for index in range(len(image_names) - 1):
            path_a = f"{config.images_dir_path}/{image_names[index]}"
            path_b = f"{config.images_dir_path}/{image_names[index + 1]}"

            if a is None:
                a = ImageSoloData(path_a)
                a.load_keypoints()

            b = ImageSoloData(path_b)
            b.load_keypoints()

            warp, certainty = self.model.match(
                a.image_path,
                b.image_path,
                device=config.device
            )

            pair: RoMa_ImagePairData = RoMa_ImagePairData(a, b)
            pair.set_warp(warp)
            pair.certainty = certainty

            pair.save_warp_certainty()

            a = b

    @staticmethod
    def get_keypoint_matches(path_a, path_b) -> RoMa_ImagePairData:
        a = ImageSoloData(path_a)
        a.load_keypoints()

        b = ImageSoloData(path_b)
        b.load_keypoints()

        pair: RoMa_ImagePairData = RoMa_ImagePairData(a, b)
        pair.load_warp_certainty()

        pair.set_left_matches_coords(a.keypoints_coords)
        right_matches_coords = pair.get_target_keypoints(a.keypoints_coords)
        pair.set_right_matches_coords(right_matches_coords)

        return pair

    @staticmethod
    def get_random_matches(path_a, path_b, confidence_threshold=0.6, num_points=5) -> RoMa_ImagePairData:
        a = ImageSoloData(path_a)
        b = ImageSoloData(path_b)

        pair: RoMa_ImagePairData = RoMa_ImagePairData(a, b)
        pair.load_warp_certainty()

        left_matches_coords = pair.get_random_reference_keypoints(confidence_threshold, num_points)
        right_matches_coords = pair.get_target_keypoints(left_matches_coords)

        pair.set_left_matches_coords(left_matches_coords)
        pair.set_right_matches_coords(right_matches_coords)

        return pair

    def show_keypoint_matches(self, path_a, path_b, num_points=5):
        """
        This function picks random keypoint pixels from image 1
        Then shows their matches from image 2
        """

        pair = self.get_keypoint_matches(path_a, path_b)
        return Visualizer.plot_matches(pair, num_points)

    def show_random_matches(self, path_a, path_b, confidence_threshold=0.6, num_points=5):
        """
        This function picks pixels from image 1 that have >= confidence_threshold.
        Then shows their matches from image 2
        """

        pair = self.get_random_matches(path_a, path_b, confidence_threshold, num_points)
        return Visualizer.plot_matches(pair, num_points)
