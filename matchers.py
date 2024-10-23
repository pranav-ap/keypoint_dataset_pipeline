from config import config
from utils import get_best_device, logger
from ImageData import Keypoints, Matches
from typing import Optional
from tqdm import tqdm
import time


class RoMaMatcher:
    def __init__(self):
        super().__init__()
        self.device = get_best_device()

        logger.info('Loading RoMaMatcher')

        from romatch import roma_outdoor
        self.model = roma_outdoor(
            device=self.device,
            upsample_res=config.image.image_shape
        )

        self.model.symmetric = False
        logger.info('Loading RoMaMatcher Done')

    def extract_warp_certainty(self, image_names):
        a: Optional[Keypoints] = None

        for name_a, name_b in tqdm(zip(image_names, image_names[1:]), desc="Extracting warps", ncols=100, total=len(image_names) - 1):
            # logger.info(f'Matcher {name_a, name_b}')
            start_time = time.time()

            if a is None:
                a = Keypoints(name_a)

            b = Keypoints(name_b)

            # Match using model and retrieve warp and certainty
            warp, certainty = self.model.match(
                a.image, b.image,
                device=self.device
            )

            # Set warp and certainty for the match pair
            pair = Matches(a, b)
            pair.set_warp(warp)
            pair.certainty = certainty
            pair.save()

            # Move forward
            a = b

            end_time = time.time()
            duration = end_time - start_time

            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)

            logger.info(f"Time taken : {int(hours)}h {int(minutes)}m {seconds:.2f}s")
