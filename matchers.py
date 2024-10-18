from config import config
from ImageData import KeypointsData, DSM_MatchesData, RoMa_MatchesData
from typing import Optional
from abc import ABC


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

    def _match_pair(self, pair: DSM_MatchesData):
        a: KeypointsData = pair.a
        b: KeypointsData = pair.b

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
        a: Optional[KeypointsData] = None

        for index in range(len(image_names) - 1):
            path_a = f"{config.images_dir_path}/{image_names[index]}"
            path_b = f"{config.images_dir_path}/{image_names[index + 1]}"

            if a is None:
                a = KeypointsData(path_a)
                a.load_keypoints()

            b = KeypointsData(path_b)
            b.load_keypoints()

            pair = DSM_MatchesData(a, b)
            self._match_pair(pair)
            pair.save_matches()

            a = b

    @staticmethod
    def show_keypoint_matches(path_a, path_b, num_points=5):
        a = KeypointsData(path_a)
        a.load_keypoints()

        b = KeypointsData(path_b)
        b.load_keypoints()

        pair = DSM_MatchesData(a, b)
        pair.load_matches()

        return pair.plot_matches(num_points)


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
        a: Optional[KeypointsData] = None

        for index in range(len(image_names) - 1):
            path_a = f"{config.images_dir_path}/{image_names[index]}"
            path_b = f"{config.images_dir_path}/{image_names[index + 1]}"

            if a is None:
                a = KeypointsData(path_a)
                a.load_keypoints()

            b = KeypointsData(path_b)
            b.load_keypoints()

            warp, certainty = self.model.match(
                a.image_path,
                b.image_path,
                device=config.device
            )

            pair: RoMa_MatchesData = RoMa_MatchesData(a, b)
            pair.set_warp(warp)
            pair.certainty = certainty

            pair.save_warp_certainty()

            a = b

    @staticmethod
    def show_keypoint_matches(path_a, path_b, confidence_threshold=0.6, num_points=None):
        """
        This function picks random keypoint pixels from image 1
        Then shows their matches from image 2
        """
        a = KeypointsData(path_a)
        a.load_keypoints()

        b = KeypointsData(path_b)
        b.load_keypoints()

        pair: RoMa_MatchesData = RoMa_MatchesData(a, b)
        pair.load_warp_certainty()

        pair.calc_keypoint_matches(confidence_threshold=confidence_threshold)

        return pair.plot_matches(num_points)

    @staticmethod
    def show_random_matches(path_a, path_b, confidence_threshold=0.6, num_points=5):
        """
        This function picks pixels from image 1 that have >= confidence_threshold.
        Then shows their matches from image 2
        """
        a = KeypointsData(path_a)
        b = KeypointsData(path_b)

        pair: RoMa_MatchesData = RoMa_MatchesData(a, b)
        pair.load_warp_certainty()

        left_matches_coords, right_matches_coords = pair.get_random_matches(
            confidence_threshold=confidence_threshold,
            num_points=num_points
        )

        dummy: RoMa_MatchesData = RoMa_MatchesData(a, b)
        dummy.set_left_matches_coords(left_matches_coords)
        dummy.set_right_matches_coords(right_matches_coords)

        return dummy.plot_matches(num_points)
