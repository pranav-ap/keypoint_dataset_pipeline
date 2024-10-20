import random
import cv2
import numpy as np
from PIL import Image
from config import config
from ImageData import KeypointsData, MatchesData


class Painter:
    """
    Contains Function to Display Keypoints & Matches
    """

    """
    Keypoints
    """

    @staticmethod
    def show_keypoints(name, num_points=None):
        kd = KeypointsData(name)
        assert kd.keypoints_coords is not None

        num_points = len(kd.keypoints_coords) if num_points is None else min(num_points, len(kd.keypoints_coords))
        keypoints = random.sample(kd.keypoints_coords, num_points)

        image_vis = cv2.drawKeypoints(
            np.array(kd.image),
            keypoints,
            None
        )

        image_vis = Image.fromarray(image_vis)
        image_vis = image_vis.resize(config.image.resize)

        return image_vis

    @staticmethod
    def show_patches(name, padding=2):
        kd = KeypointsData(name)
        assert kd.keypoints_coords is not None

        num_rows, num_cols = kd.grid_patches_shape

        # Calculate the size of the output image
        patch_height, patch_width = kd.grid_patches[0, 0].size
        grid_width = num_cols * patch_width + (num_cols - 1) * padding
        grid_height = num_rows * patch_height + (num_rows - 1) * padding

        # Create a blank canvas for the grid
        grid_image = Image.new(
            'RGB',
            (grid_width, grid_height),
            color=(255, 255, 255)
        )

        for i in range(num_rows):
            for j in range(num_cols):
                # Calculate the position to paste the patch
                x = j * (patch_width + padding)
                y = i * (patch_height + padding)

                # Paste the patch at the calculated position
                grid_image.paste(kd.grid_patches[i, j], (x, y))

        return grid_image

    """
    Matches
    """

    @staticmethod
    def show_keypoint_matches(name_a, name_b, confidence_threshold=0.6, num_points=None):
        """
        This function picks random keypoint pixels from image 1
        Then shows their matches from image 2
        """
        pair = MatchesData.load_from_names(name_a, name_b)

        assert pair.a.image is not None
        assert pair.b.image is not None

        left_matches_coords, right_matches_coords = pair.get_good_matches(
            pair.a.keypoints_coords,
            confidence_threshold
        )

        num_points = len(left_matches_coords) if num_points is None else min(num_points, len(left_matches_coords))
        matches = [cv2.DMatch(idx, idx, 0.) for idx in range(num_points)]

        image_vis = cv2.drawMatches(
            np.array(pair.a.image), left_matches_coords,
            np.array(pair.b.image), right_matches_coords,
            matches,
            outImg=None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        image_vis = Image.fromarray(image_vis)
        image_vis = image_vis.resize((pair.a.image.width * 2, pair.a.image.height))

        return image_vis

    @staticmethod
    def show_filtered_keypoint_matches(name_a, name_b, num_points=None):
        """
        This function picks random keypoint pixels from image 1
        Then shows their matches from image 2
        """
        pair = MatchesData.load_from_names(name_a, name_b, load_filtered_matches=True)

        assert pair.a.image is not None
        assert pair.b.image is not None
        assert pair.left_matches_coords_filtered is not None
        assert pair.right_matches_coords_filtered is not None

        num_points = len(pair.left_matches_coords_filtered) if num_points is None else min(num_points, len(pair.left_matches_coords_filtered))
        matches = [cv2.DMatch(idx, idx, 0.) for idx in range(num_points)]

        image_vis = cv2.drawMatches(
            np.array(pair.a.image), pair.left_matches_coords_filtered,
            np.array(pair.b.image), pair.right_matches_coords_filtered,
            matches,
            outImg=None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        image_vis = Image.fromarray(image_vis)
        image_vis = image_vis.resize((pair.a.image.width * 2, pair.a.image.height))

        return image_vis

    @staticmethod
    def show_random_matches(name_a, name_b, confidence_threshold=0.6, num_points=None):
        pair = MatchesData.load_from_names(name_a, name_b)

        assert pair.a.image is not None
        assert pair.b.image is not None

        left_matches_coords = pair.get_random_reference_keypoints(confidence_threshold, num_points)
        left_matches_coords, right_matches_coords = pair.get_good_matches(left_matches_coords, confidence_threshold)

        num_points = len(left_matches_coords) if num_points is None else min(num_points, len(left_matches_coords))
        matches = [cv2.DMatch(idx, idx, 0.) for idx in range(num_points)]

        image_vis = cv2.drawMatches(
            np.array(pair.a.image), left_matches_coords,
            np.array(pair.b.image), right_matches_coords,
            matches,
            outImg=None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        image_vis = Image.fromarray(image_vis)
        image_vis = image_vis.resize((pair.a.image.width * 2, pair.a.image.height))

        return image_vis
