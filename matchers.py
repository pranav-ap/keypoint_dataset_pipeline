from config import config
from utils import get_best_device, logger
from ImageData import KeypointsData, MatchesData
from typing import Optional
from rich.progress import Progress


class RoMaMatcher:
    def __init__(self):
        super().__init__()
        self.device = get_best_device()

        logger.info('Loading RoMaMatcher')

        from romatch import roma_outdoor
        self.model = roma_outdoor(
            device=self.device,
            coarse_res=560,
            upsample_res=config.image.resize
        )

        self.model.symmetric = False
        logger.info('Loading RoMaMatcher Done')

    def extract_warp_certainty(self, image_names):
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Extracting warps...",
                total=len(image_names) - 1
            )

            a: Optional[KeypointsData] = None

            for name_a, name_b in zip(image_names, image_names[1:]):
                if a is None:
                    a = KeypointsData(name_a)
                    a.load()

                b = KeypointsData(name_b)
                b.load()

                # Match using model and retrieve warp and certainty
                warp, certainty = self.model.match(
                    a.image_path, b.image_path,
                    device=self.device
                )

                # Set warp and certainty for the match pair
                pair = MatchesData(a, b)
                pair.set_warp(warp)
                pair.certainty = certainty
                pair.save()

                # Move forward
                a = b

                progress.advance(task)

            progress.stop()
