from config import config
from utils import logger, make_clear_directory
import random
import cv2
import numpy as np
from PIL import Image
from ImageData import Keypoints, Matches, DataStore


class Painter:
    def __init__(self):
        self.data_store = DataStore(mode='r')

    def _resize_image(self, image, size):
        return Image.fromarray(image).resize(size)

    def _draw_keypoints(self, image, keypoints_coords, num_points):
        keypoints = random.sample(keypoints_coords, num_points)
        return cv2.drawKeypoints(np.array(image), keypoints, None)

    def _draw_matches(self, pair, left_coords, right_coords, num_points):
        matches = [cv2.DMatch(idx, idx, 0.) for idx in range(num_points)]
        return cv2.drawMatches(
            np.array(pair.a.image), left_coords,
            np.array(pair.b.image), right_coords,
            matches, outImg=None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

    def show_patches(self, name, padding=2):
        kd = Keypoints.load_from_name(name, self.data_store)

        num_rows, num_cols = kd.patches_shape
        print(f'num_rows, num_cols {num_rows, num_cols}')
        patch_height, patch_width = kd.patch_images[0, 0].size
        print(f'patch_height, patch_width {patch_height, patch_width}')

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

    def _show_keypoints(self, kd, num_points, keypoint_attr, resize_size):
        coords = getattr(kd, keypoint_attr).as_image_coords()
        assert coords is not None

        num_points = len(coords) if num_points is None else min(num_points, len(coords))
        logger.info(f'Number of Keypoints {num_points}')
        image_vis = self._draw_keypoints(kd.image, coords, num_points)

        return self._resize_image(image_vis, resize_size)

    def show_keypoints(self, name, level='both', filtered=False, num_points=None):
        kd = Keypoints.load_from_name(name, self.data_store, is_filtered=filtered)

        if level == 'image':
            keypoint_attr = 'image_keypoints_filtered' if filtered else 'image_keypoints'
            return self._show_keypoints(kd, num_points, keypoint_attr, config.image.image_shape)
        elif level == 'patch':
            keypoint_attr = 'patches_keypoints_filtered' if filtered else 'patches_keypoints'
            return self._show_keypoints(kd, num_points, keypoint_attr, config.image.image_shape)

        coords = kd.get_all_coords() if not filtered else kd.get_all_filtered_coords()
        assert coords is not None

        num_points = len(coords) if num_points is None else min(num_points, len(coords))
        logger.info(f'Number of Keypoints {num_points}')

        image_vis = self._draw_keypoints(kd.image, coords, num_points)

        return self._resize_image(image_vis, config.image.image_shape)

    """
    Matches Display Functions
    """

    def _show_matches(self, pair, left_coords, right_coords, num_points):
        image_vis = self._draw_matches(pair, left_coords, right_coords, num_points)
        return self._resize_image(image_vis, (pair.a.image.width * 2, pair.a.image.height))

    def show_matches(self, name_a, name_b, confidence_threshold=0.6, num_points=None):
        pair = Matches.load_from_names(name_a, name_b, self.data_store)
        coords = pair.a.image_keypoints.as_image_coords()
        left_coords, right_coords = pair.get_good_matches(coords, confidence_threshold)
        num_points = len(left_coords) if num_points is None else min(num_points, len(left_coords))
        logger.info(f'Number of Matches {num_points}')

        return self._show_matches(pair, left_coords, right_coords, num_points)

    def show_filtered_matches(self, name_a, name_b, num_points=None):
        pair = Matches.load_from_names(name_a, name_b, self.data_store, load_coords=True)
        num_points = len(pair.left_coords) if num_points is None else min(num_points, len(pair.left_coords))
        logger.info(f'Number of Matches {num_points}')

        return self._show_matches(pair, pair.left_coords, pair.right_coords, num_points)

    """
    Original Image Functions
    """
    
    def show_matches_on_original_image(self, name_a, name_b):
        current_width, current_height = config.image.image_shape
        new_width, new_height = config.image.original_image_shape

        def resize_coordinates(x, y):
            new_x = x * (new_width / current_width)
            new_y = y * (new_height / current_height)
            kp = cv2.KeyPoint(round(new_x), round(new_y), 1)
            return kp
            
        pair = Matches.load_from_names(name_a, name_b, self.data_store, load_coords=True, no_patches=True, must_resize=False)
        num_points = len(pair.left_coords)
        logger.info(f'Number of Matches {num_points}')

        left_coords = [resize_coordinates(kp.pt[0], kp.pt[1]) for kp in pair.left_coords]
        right_coords = [resize_coordinates(kp.pt[0], kp.pt[1]) for kp in pair.right_coords]

        return self._show_matches(pair, left_coords, right_coords, num_points)

