import cv2
from config import config
from ImageData import ImageSoloData, ImagePairData
import numpy as np
import torch
from PIL import Image
from typing import Optional
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

        config.IMAGE_RESIZE = (784, 784)

        from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
        self.matcher = DualSoftMaxMatcher()

    """
    Visualization
    """

    @staticmethod
    def show_all_matches(path_a, path_b, num_points=10):
        a = ImageSoloData(path_a, resize=config.IMAGE_RESIZE)
        a.load_keypoints()

        b = ImageSoloData(path_b, resize=config.IMAGE_RESIZE)
        b.load_keypoints()

        pair = ImagePairData(a, b)
        pair.load_matches()

        a, b = np.array(a.image), np.array(b.image)

        left_matches = [
            cv2.KeyPoint(x, y, 1.)
            for x, y in pair.left_matches
        ]

        right_matches = [
            cv2.KeyPoint(x, y, 1.)
            for x, y in pair.right_matches
        ]

        num_points = len(left_matches) if num_points is None else min(num_points, len(left_matches))
        matches = [cv2.DMatch(idx, idx, 0.) for idx in range(num_points)]

        image_vis = cv2.drawMatches(
            a, left_matches,
            b, right_matches,
            matches,
            outImg=None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        image_vis = Image.fromarray(image_vis)

        return image_vis

    """
    Match
    """

    def _match_pair(self, pair: ImagePairData):
        a: ImageSoloData = pair.a
        b: ImageSoloData = pair.b

        matches_A, matches_B, batch_ids = self.matcher.match(
            torch.tensor(a.keypoints), torch.tensor(a.descriptions),
            torch.tensor(b.keypoints), torch.tensor(b.descriptions),
            normalize=True,
            inv_temp=20,
            threshold=0.1
        )

        pair.left_matches, pair.right_matches = self.matcher.to_pixel_coords(
            matches_A, matches_B,
            a.H, a.W, b.H, b.W
        )

    def extract_matches(self, image_names):
        a: Optional[ImageSoloData] = None

        for index in range(len(image_names) - 1):
            path_a = f"{config.images_dir_path}/{image_names[index]}"
            path_b = f"{config.images_dir_path}/{image_names[index + 1]}"

            if a is None:
                a = ImageSoloData(path_a, resize=config.IMAGE_RESIZE)
                a.load_keypoints()

            b = ImageSoloData(path_b, resize=config.IMAGE_RESIZE)
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
            upsample_res=(864, 1152)
        )

        self.model.symmetric = False
        self.H, self.W = self.model.get_output_resolution()

        config.IMAGE_RESIZE = (self.W, self.H)  # (1152, 864)

    """
    Utils
    """

    def _get_corresponding_points(self, points_in_im1, warp):
        points_in_im2 = []

        for pt in points_in_im1:
            x_im1, y_im1 = pt

            # Cast to integers
            x_im1, y_im1 = int(x_im1), int(y_im1)

            w = warp[y_im1, x_im1]
            A, B = self.model.to_pixel_coordinates(w, self.H, self.W, self.H, self.W)
            x_im2, y_im2 = B

            points_in_im2.append((int(x_im2.item()), int(y_im2.item())))

        return points_in_im2

    @staticmethod
    def _get_points_from_certainty(certainty, threshold=0.6, num_points=5):
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

        points = [(x_coords[i], y_coords[i]) for i in indices]

        return points

    """
    Visualize
    """

    @staticmethod
    def _plot_matches(pair: ImagePairData, points_in_im1, points_in_im2, num_points):
        left_matches = [cv2.KeyPoint(int(x), int(y), 1.) for x, y in points_in_im1]
        right_matches = [cv2.KeyPoint(int(x), int(y), 1.) for x, y in points_in_im2]

        assert len(left_matches) == len(right_matches)

        num_points = len(left_matches) if num_points is None else min(num_points, len(left_matches))
        matches = [cv2.DMatch(idx, idx, 0.) for idx in range(num_points)]

        a, b = np.array(pair.a.image), np.array(pair.b.image)

        image_vis = cv2.drawMatches(
            a, left_matches,
            b, right_matches,
            matches,
            outImg=None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        image_vis = Image.fromarray(image_vis)
        return image_vis

    def extract_and_show_random_matches(self, path_a, path_b, confidence_threshold=0.6, num_points=5):
        """
        This function picks pixels from image 1 that have >= confidence_threshold.
        Then shows their matches from image 2
        """

        a = ImageSoloData(path_a, resize=config.IMAGE_RESIZE)
        b = ImageSoloData(path_b, resize=config.IMAGE_RESIZE)
        pair = ImagePairData(a, b)

        warp, certainty = self._match_pair(pair)

        points_in_im1 = self._get_points_from_certainty(certainty, confidence_threshold, num_points)
        points_in_im2 = self._get_corresponding_points(points_in_im1, warp)

        return self._plot_matches(pair, points_in_im1, points_in_im2, num_points)

    def extract_and_show_matches_from_keypoints(self, path_a, path_b, num_points=5):
        """
        This function picks random keypoint pixels from image 1
        Then shows their matches from image 2
        """

        a = ImageSoloData(path_a, resize=config.IMAGE_RESIZE)
        a.load_keypoints()

        b = ImageSoloData(path_b, resize=config.IMAGE_RESIZE)
        b.load_keypoints()

        pair = ImagePairData(a, b)

        warp, certainty = self._match_pair(pair)

        points_in_im1 = a.keypoints.cpu().numpy() if isinstance(a.keypoints, torch.Tensor) else a.keypoints
        points_in_im1 = points_in_im1.squeeze(0)
        num_points = len(points_in_im1) if num_points is None else min(num_points, len(points_in_im1))

        width, height = a.image.size

        points_in_im1 = [
            ((x + 1) * (width / 2), (y + 1) * (height / 2))
            for x, y in points_in_im1
        ]

        points_in_im2 = self._get_corresponding_points(points_in_im1, warp)

        return self._plot_matches(pair, points_in_im1, points_in_im2, num_points)

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
                a = ImageSoloData(path_a, resize=config.IMAGE_RESIZE)
                a.load_keypoints()

            b = ImageSoloData(path_b, resize=config.IMAGE_RESIZE)
            b.load_keypoints()

            pair = ImagePairData(a, b)

            warp, certainty = self._match_pair(pair)

            points_in_im1 = a.keypoints.cpu().numpy() if isinstance(a.keypoints, torch.Tensor) else a.keypoints
            points_in_im1 = points_in_im1.squeeze(0)

            width, height = a.image.size

            points_in_im1 = [
                ((x + 1) * (width / 2), (y + 1) * (height / 2))
                for x, y in points_in_im1
            ]

            points_in_im2 = self._get_corresponding_points(points_in_im1, warp)

            pair.left_matches = np.array(points_in_im1)
            pair.right_matches = np.array(points_in_im2)

            pair.save_matches()
            a = b
