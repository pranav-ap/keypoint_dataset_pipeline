from abc import ABC
from typing import Optional

from tqdm import tqdm

from config import config
from utils import get_best_device, logger
from .ImageData import Keypoints, Matches


class KeypointMatcher(ABC):
    device = get_best_device()

    def __init__(self, data_store):
        self.data_store = data_store


class RoMaMatcher(KeypointMatcher):
    def __init__(self, data_store):
        super().__init__(data_store)

        logger.info('Loading RoMaMatcher')
        from romatch import roma_outdoor
        self.model = roma_outdoor(
            device=self.device,
            # (height, width)
            upsample_res=(config.image.crop_image_shape[1], config.image.crop_image_shape[0])
        )

        self.model.symmetric = False
        logger.info('Loading RoMaMatcher Done')

    def extract_warp_certainty(self, image_names):
        a: Optional[Keypoints] = None

        for name_a, name_b in tqdm(zip(image_names, image_names[1:]), desc="Extracting warps", ncols=100,
                                   total=len(image_names) - 1):
            if a is None:
                a = Keypoints(name_a, self.data_store)

            b = Keypoints(name_b, self.data_store)

            # Match using model and retrieve warp and certainty
            warp, certainty = self.model.match(
                a.image, b.image,
                device=self.device
            )

            # Set warp and certainty for the match pair
            pair = Matches(a, b, self.data_store)
            pair.set_warp(warp)
            pair.certainty = certainty
            pair.save()

            # Move forward
            a = b
