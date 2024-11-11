import cv2
import numpy as np
from PIL import Image, ImageDraw

from config import config
from utils import logger
from .DataStore import DataStore
from .ImageData import Keypoints, Matches


class Painter:
    def __init__(self):
        if config.task.consider_samples:
            config.paths[config.task.name].output = config.paths.samples.output

        self.data_store = DataStore(mode='r')
        self.data_store.init()

    @staticmethod
    def _to_image(image):
        return Image.fromarray(image)

    @staticmethod
    def _draw_keypoints(image, keypoints_coords):
        # noinspection PyTypeChecker
        return cv2.drawKeypoints(np.array(image), keypoints_coords, None)

    @staticmethod
    def _draw_matches(pair, left_coords, right_coords):
        num_points = len(left_coords)
        matches = [cv2.DMatch(idx, idx, 0.) for idx in range(num_points)]
        # noinspection PyTypeChecker
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

    def _show_keypoints(self, kd, keypoint_attr):
        coords = getattr(kd, keypoint_attr).as_image_coords()
        assert coords is not None

        num_points = len(coords)
        logger.info(f'Number of Keypoints {num_points}')
        image_vis = self._draw_keypoints(kd.image, coords)

        return self._to_image(image_vis)

    def show_keypoints(self, name, level='both', filtered=False):
        kd = Keypoints.load_from_name(name, self.data_store, is_filtered=filtered)

        if level == 'image':
            keypoint_attr = 'image_keypoints_filtered' if filtered else 'image_keypoints'
            return self._show_keypoints(kd, keypoint_attr)
        elif level == 'patch':
            keypoint_attr = 'patches_keypoints_filtered' if filtered else 'patches_keypoints'
            return self._show_keypoints(kd, keypoint_attr)

        coords = kd.get_all_coords() if not filtered else kd.get_all_filtered_coords()
        assert coords is not None

        num_points = len(coords)
        logger.info(f'Number of Keypoints {num_points}')

        image_vis = self._draw_keypoints(kd.image, coords)

        return self._to_image(image_vis)

    """
    Matches Display Functions
    """

    def _show_matches(self, pair, left_coords, right_coords):
        image_vis = self._draw_matches(pair, left_coords, right_coords)
        image_vis = self._to_image(image_vis)

        draw = ImageDraw.Draw(image_vis)
        draw.line((image_vis.width // 2, 0, image_vis.width // 2, image_vis.height), fill="white", width=2)

        return image_vis

    def show_matches(self, name_a, name_b):
        pair = Matches.load_from_names(name_a, name_b, self.data_store)
        coords = pair.a.image_keypoints.as_image_coords()
        confidence_threshold = config.roma.filter.confidence_threshold
        left_coords, right_coords = pair.get_good_matches(coords, confidence_threshold)
        num_points = len(left_coords)
        logger.info(f'Number of Matches {num_points}')

        return self._show_matches(pair, left_coords, right_coords)

    def show_filtered_matches(self, name_a, name_b):
        pair = Matches.load_from_names(name_a, name_b, self.data_store, load_coords=True)
        num_points = len(pair.small_left_coords)
        logger.info(f'Number of Matches {num_points}')

        return self._show_matches(pair, pair.small_left_coords, pair.small_right_coords)

    def show_matches_on_original_image(self, name_a, name_b):
        pair = Matches.load_from_names(name_a, name_b, self.data_store, load_coords=True)
        num_points = len(pair.original_left_coords)
        logger.info(f'Number of Matches {num_points}')

        return self._show_matches(pair, pair.original_left_coords, pair.original_right_coords)
