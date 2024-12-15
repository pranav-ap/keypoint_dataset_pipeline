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

    def _show_keypoints(self, kd, keypoint_attr):
        coords = getattr(kd, keypoint_attr).as_image_coords()
        assert coords is not None

        num_points = len(coords)
        logger.info(f'Number of Keypoints {num_points}')
        image_vis = self._draw_keypoints(kd.image, coords)

        return self._to_image(image_vis)

    def show_keypoints(self, name, filtered=False):
        kd = Keypoints.load_from_name(name, self.data_store, is_filtered=filtered)
        coords = kd.get_all_coords() if not filtered else kd.get_all_filtered_coords()
        assert coords is not None

        num_points = len(coords)
        logger.info(f'Number of Keypoints {num_points}')

        image_vis = self._draw_keypoints(kd.image, coords)

        return self._to_image(image_vis)

    @staticmethod
    def _draw_matches(image_a, image_b, left_coords, right_coords):
        num_points = len(left_coords)
        matches = [cv2.DMatch(idx, idx, 0.) for idx in range(num_points)]
        # noinspection PyTypeChecker
        return cv2.drawMatches(
            np.array(image_a), left_coords,
            np.array(image_b), right_coords,
            matches, outImg=None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

    def _show_matches(self, pair, left_coords, right_coords, on_original_image=False):
        image_a = pair.a.original_image if on_original_image else pair.a.image
        image_b = pair.b.original_image if on_original_image else pair.b.image

        image_vis = self._draw_matches(image_a, image_b, left_coords, right_coords)
        image_vis = self._to_image(image_vis)

        draw = ImageDraw.Draw(image_vis)
        # draw.line((image_vis.width / 2, 0, image_vis.width / 2, image_vis.height), fill="white", width=2)

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
        num_points = len(pair.reference_crop_coords)
        logger.info(f'Number of Matches {num_points}')

        return self._show_matches(pair, pair.reference_crop_coords, pair.target_crop_coords)

    def show_matches_on_original_image(self, name_a, name_b):
        pair: Matches = Matches.load_from_names(name_a, name_b, self.data_store, load_coords=True)

        left_coords, right_coords = pair.get_coords_on_original_image()
        num_points = len(left_coords)
        logger.info(f'Number of Matches {num_points}')

        return self._show_matches(pair, left_coords, right_coords)
