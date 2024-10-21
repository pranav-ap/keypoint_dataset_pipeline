from config import config
from utils import logger
import random
import cv2
import numpy as np
from PIL import Image
from ImageData import KeypointsData, MatchesData


class Painter:
    """
    Contains functions to display keypoints & matches.
    """

    @staticmethod
    def _resize_image(image, size):
        return Image.fromarray(image).resize(size)

    @staticmethod
    def _draw_keypoints(image, keypoints_coords, num_points):
        keypoints = random.sample(keypoints_coords, num_points)
        return cv2.drawKeypoints(np.array(image), keypoints, None)

    @staticmethod
    def _draw_matches(pair, left_coords, right_coords, num_points):
        matches = [cv2.DMatch(idx, idx, 0.) for idx in range(num_points)]
        return cv2.drawMatches(
            np.array(pair.a.image), left_coords,
            np.array(pair.b.image), right_coords,
            matches, outImg=None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

    @staticmethod
    def show_patches(name, also_filtered=False, padding=2):
        kd = KeypointsData.load_from_name(name, also_filtered)

        num_rows, num_cols = kd.patches_shape
        patch_height, patch_width = kd.patch_images[0, 0].size
        grid_size = (
            num_cols * patch_width + (num_cols - 1) * padding,
            num_rows * patch_height + (num_rows - 1) * padding
        )

        grid_image = Image.new('RGB', grid_size, color=(255, 255, 255))

        for i in range(num_rows):
            for j in range(num_cols):
                x, y = j * (patch_width + padding), i * (patch_height + padding)
                grid_image.paste(kd.patch_images[i, j], (x, y))

        return grid_image

    """
    Keypoints Display Functions
    """

    @staticmethod
    def _show_keypoints(kd, num_points, keypoint_attr, resize_size):
        coords = getattr(kd, keypoint_attr).as_image_coords()
        assert coords is not None

        num_points = len(coords) if num_points is None else min(num_points, len(coords))
        image_vis = Painter._draw_keypoints(kd.image, coords, num_points)

        return Painter._resize_image(image_vis, resize_size)

    @staticmethod
    def show_keypoints(name, level='both', filtered=False, num_points=None):
        kd = KeypointsData.load_from_name(name, also_filtered=filtered)

        if level == 'image':
            keypoint_attr = 'image_keypoints_filtered' if filtered else 'image_keypoints'
            return Painter._show_keypoints(kd, num_points, keypoint_attr, config.image.resize)
        elif level == 'patch':
            keypoint_attr = 'patches_keypoints_filtered' if filtered else 'patches_keypoints'
            return Painter._show_keypoints(kd, num_points, keypoint_attr, config.image.resize)

        coords = kd.get_all_coords() if not filtered else kd.get_all_filtered_coords()
        assert coords is not None

        num_points = len(coords) if num_points is None else min(num_points, len(coords))
        image_vis = Painter._draw_keypoints(kd.image, coords, num_points)

        return Painter._resize_image(image_vis, config.image.resize)

    """
    Matches Display Functions
    """

    @staticmethod
    def _show_matches(pair, left_coords, right_coords, num_points):
        image_vis = Painter._draw_matches(pair, left_coords, right_coords, num_points)
        return Painter._resize_image(image_vis, (pair.a.image.width * 2, pair.a.image.height))

    @staticmethod
    def show_matches(name_a, name_b, confidence_threshold=0.6, num_points=None):
        pair = MatchesData.load_from_names(name_a, name_b)
        coords = pair.a.image_keypoints.as_image_coords()
        left_coords, right_coords = pair.get_good_matches(coords, confidence_threshold)
        num_points = len(left_coords) if num_points is None else min(num_points, len(left_coords))

        return Painter._show_matches(pair, left_coords, right_coords, num_points)

    @staticmethod
    def show_filtered_matches(name_a, name_b, num_points=None):
        pair = MatchesData.load_from_names(name_a, name_b, load_coords=True)
        num_points = len(pair.left_coords) if num_points is None else min(num_points, len(pair.left_coords))

        return Painter._show_matches(pair, pair.left_coords, pair.right_coords, num_points)
