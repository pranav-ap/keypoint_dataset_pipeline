import cv2
from config import config
from ImageData import ImageSoloData, ImagePairData
from visualizer import Visualizer
import numpy as np
import torch
from typing import List, Optional
from abc import ABC, abstractmethod


class KeypointMatcher(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def extract_matches(self, image_names):
        pass


class DeDoDeMatcher(KeypointMatcher):
    def __init__(self):
        super().__init__()

        from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
        self.matcher = DualSoftMaxMatcher()

    """
    Match
    """

    def _match_pair(self, pair: ImagePairData):
        a: ImageSoloData = pair.a
        b: ImageSoloData = pair.b

        matches_A, matches_B, batch_ids = self.matcher.match(
            a.keypoints, a.descriptions,
            b.keypoints, b.descriptions,
            normalize=True,
            inv_temp=20,
            threshold=0.1
        )

        """
        > matches_A
        tensor([[ 0.9809,  0.7462],
                [ 0.7054,  0.7487],
                [-0.1314,  0.5293],
                ...,
                [-0.2793,  0.3482],
                [ 0.4630,  0.8074],
                [ 0.6901, -0.2921]], device='cuda:0')
        """

        H = a.image.height
        W = a.image.width

        pair.left_matches, pair.right_matches = self.matcher.to_pixel_coords(
            matches_A, matches_B,
            H, W, H, W
        )

        """
        > left_matches
        tensor([[776.5000, 684.4999],
                [668.5000, 685.5000],
                [340.5000, 599.5000],
                ...,
                [282.5000, 528.5000],
                [573.5000, 708.5000],
                [662.5000, 277.5000]], device='cuda:0')
        """

        pair.left_matches_coords = [
            cv2.KeyPoint(int(x.item()), int(y.item()), 1.)
            for x, y in pair.left_matches
        ]

        pair.right_matches_coords = [
            cv2.KeyPoint(int(x.item()), int(y.item()), 1.)
            for x, y in pair.right_matches
        ]

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

            pair = ImagePairData(a, b)

            self._match_pair(pair)
            pair.save_matches()

            a = b


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

    """
    Utils
    """

    def _get_corresponding_keypoint_coords(self, a: ImageSoloData, warp) -> List[cv2.KeyPoint]:
        b_keypoint_coords = []
        H, W = a.image.height, a.image.width

        for pt in a.keypoints_coords:
            x_a, y_a = pt.pt
            x_a, y_a = int(x_a), int(y_a)

            w = warp[y_a, x_a]
            A, B = self.model.to_pixel_coordinates(w, H, W, H, W)

            x_b, y_b = B
            x_b, y_b = int(x_b.item()), int(y_b.item())

            b_keypoint_coords.append(cv2.KeyPoint(x_b, y_b, 1.))

        return b_keypoint_coords

    @staticmethod
    def _get_points_from_certainty(certainty, threshold=0.6, num_points=5) -> List[cv2.KeyPoint]:
        certainty_cpu = certainty.cpu().numpy() if isinstance(certainty, torch.Tensor) else certainty
        # Convert certainty to a binary mask where values are above the threshold
        mask = certainty_cpu > threshold

        # Get the coordinates of the points where mask is True
        y_coords, x_coords = np.where(mask)

        # Randomly sample from the points above the threshold
        if len(y_coords) > num_points:
            indices = np.random.choice(len(y_coords), size=num_points, replace=False)
        else:
            indices = np.arange(len(y_coords))  # Take all if not enough points

        points = [
            cv2.KeyPoint(x_coords[i], y_coords[i], 1.)
            for i in indices
        ]

        return points

    """
    Visualize
    """

    def extract_and_show_random_matches(self, path_a, path_b, confidence_threshold=0.6, num_points=5):
        """
        This function picks pixels from image 1 that have >= confidence_threshold.
        Then shows their matches from image 2
        """

        a = ImageSoloData(path_a)
        b = ImageSoloData(path_b)
        pair = ImagePairData(a, b)

        warp, certainty = self._match_pair(pair)

        points_in_im1 = self._get_points_from_certainty(certainty, confidence_threshold, num_points)
        points_in_im2 = self._get_corresponding_keypoint_coords(a, warp)

        pair.left_matches_coords = points_in_im1
        pair.right_matches_coords = points_in_im2

        return Visualizer.plot_matches(pair, num_points)

    def extract_and_show_matches_from_keypoints(self, path_a, path_b, num_points=5):
        """
        This function picks random keypoint pixels from image 1
        Then shows their matches from image 2
        """

        a = ImageSoloData(path_a)
        a.load_keypoints()

        b = ImageSoloData(path_b)
        b.load_keypoints()

        pair = ImagePairData(a, b)

        warp, certainty = self._match_pair(pair)

        points_in_im1 = a.keypoints_coords
        points_in_im2 = self._get_corresponding_keypoint_coords(a, warp)

        pair.left_matches_coords = points_in_im1
        pair.right_matches_coords = points_in_im2

        return Visualizer.plot_matches(pair, num_points)

    """
    Match
    """

    def _match_pair(self, pair: ImagePairData):
        warp, certainty = self.model.match(
            pair.a.image_path,
            pair.b.image_path,
            device=config.device
        )

        return warp, certainty

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

            pair = ImagePairData(a, b)

            warp, certainty = self._match_pair(pair)

            points_in_im2 = self._get_corresponding_keypoint_coords(a, warp)

            pair.left_matches = a.keypoints_coords
            pair.right_matches = points_in_im2

            pair.save_matches()
            a = b
