import sys

print(f"{sys.path=}")

try:
    print(">>> Importing")

    import ipdb
    import numpy as np
    from PIL import Image
    import h5py

    import torch

    from esda.moran import Moran
    from libpysal.weights import lat2W

    from track_utils import (
        crop_image_alb,
        crop_image,
        show_batch,
        min_max_normalize,
        filter_bad_reference_keypoints,
    )

    print(">>> Imported")
except Exception as e:
    print(">>> Failed to import", e)


TIME_OFFSETS = {
    # MG
    "MGO01_low_light": 200175,
    "MGO02_hand_puncher": 255186,
    "MGO03_hand_shooter_easy": 43666,
    "MGO04_hand_shooter_hard": 226375,
    "MGO05_inspect_easy": -117270,
    "MGO06_inspect_hard": 410814,
    "MGO07_mapping_easy": 191231,
    "MGO08_mapping_hard": 261838,
    "MGO09_short_1_updown": -583624,
    "MGO10_short_2_panorama": -862202,
    "MGO11_short_3_backandforth": 590597,
    # MO
    "MOO01_hand_puncher_1": -371496,
    "MOO02_hand_puncher_2": -1245475,
    "MOO03_hand_shooter_easy": -83872,
    "MOO04_hand_shooter_hard": -2250703,
    "MOO05_inspect_easy": -1874999,
    "MOO06_inspect_hard": -1626959,
    "MOO07_mapping_easy": -2147866,
    "MOO08_mapping_hard": -4276950,
    "MOO09_short_1_updown": -3607584,
    "MOO10_short_2_panorama": -1030205,
    "MOO11_short_3_backandforth": -1295536,
}

DATASET = "MOO01_hand_puncher_1"
TIME_OFFSET = TIME_OFFSETS[DATASET]
DATASET_PATH = f"E:/thesis_code/datasets/output/output_all/basalt/monado_slam/{DATASET}"


class Reader:
    def __init__(self, cam: int):
        dataset = DATASET
        folder_path = DATASET_PATH
        filename = "data.hdf5"  # do not change
        filepath = f"{folder_path}/{filename}"
        self._file = h5py.File(filepath, "r")

        self.cam = f"cam{cam}"
        assert self.cam in ["cam0", "cam1"]

        # [width, height]
        w = 640
        h = 480
        r = 14
        cw = w // r * r
        ch = h // r * r
        print(f"{cw=}, {ch=}")
        hpad = (w - cw) // 2
        vpad = (h - ch) // 2
        assert cw + hpad * 2 == w and ch + vpad * 2 == h

        self.original_image_shape = [w, h]
        self.crop_image_shape = [cw, ch]
        self.pad = [hpad, vpad]

        self._init_groups_read_mode()

    def _init_groups_read_mode(self):
        self._detector = self._file[f"{self.cam}/detector"]
        self._matcher = self._file[f"{self.cam}/matcher"]
        self._filter = self._file[f"{self.cam}/filter"]
        self._matches = self._file[f"{self.cam}/matches"]

        self.detector_normalised = self._detector["normalised"]
        self.detector_confidences = self._detector["confidences"]

        self.matcher_warp = self._matcher["warp"]  # only one you will need
        self.matcher_certainty = self._matcher["certainty"]

        self.filter_normalised = self._filter["normalised"]
        self.filter_confidences = self._filter["confidences"]

        self.cropped_image_reference_coords = self._matches["crop/reference_coords"]
        self.cropped_image_target_coords = self._matches["crop/target_coords"]

    def close(self):
        self._file.close()

    def _warp_to_pixel_coords(self, warp):
        """
        This function is from a RoMa utils file
        """
        h1, w1 = 476, 630
        h2, w2 = 476, 630

        warp1 = warp[..., :2]
        warp1 = torch.stack(
            (
                w1 * (warp1[..., 0] + 1) / 2,
                h1 * (warp1[..., 1] + 1) / 2,
            ),
            axis=-1,
        )

        warp2 = warp[..., 2:]
        warp2 = torch.stack(
            (
                w2 * (warp2[..., 0] + 1) / 2,
                h2 * (warp2[..., 1] + 1) / 2,
            ),
            axis=-1,
        )

        return torch.cat((warp1, warp2), dim=-1)

    def load_warp(self, pair_name):
        warp = self.matcher_warp[pair_name][()]
        warp = torch.from_numpy(warp)

        pixel_coords = self._warp_to_pixel_coords(warp)
        certainty = self.matcher_certainty[pair_name][()]

        return pixel_coords, certainty

    def get_target_keypoint(self, flow, certainties, x0, y0):
        hpad, vpad = self.pad
        cw, ch = self.crop_image_shape

        if x0 < 0 or x0 > cw or y0 < 0 or y0 > ch:
            return 0, 0, 0

        # global left_image
        # if get_stddev_im(left_image, x=x0, y=y0, patch_size=32) > 5:
        #     return 0, 0, 0

        x0 -= hpad
        y0 -= vpad

        # we int() instead of round() since roma estimates for pixel center and not pixel border
        y0 = int(y0)
        x0 = int(x0)

        _, _, x1, y1 = flow[y0, x0]
        x1, y1 = x1.item(), y1.item()

        certainty = certainties[y0, x0]

        x1 += hpad
        y1 += vpad

        return x1, y1, certainty

    def close(self):
        self._file.close()

    def print_hdf5_structure(self):
        def print_group(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")

        self._file.visititems(print_group)


reader0 = Reader(0)
reader1 = Reader(1)

readers = [reader0, reader1]
curr_reader = None
curr_flow = None
curr_certs = None

left_image = None


def track_set_pair(cam0: int, ts0: int, cam1: int, ts1: int):
    assert cam0 == cam1, "The dataset does not provide inter-camera matches"

    global curr_reader, curr_flow, curr_certs
    ts0 -= TIME_OFFSET
    ts1 -= TIME_OFFSET

    curr_reader = readers[cam0]

    pair_name = f"{ts0}_{ts1}"
    curr_flow, curr_certs = curr_reader.load_warp(pair_name)

    global left_image
    filepath = (
        f"E:/thesis_code/datasets/monado_slam/{DATASET}/mav0/cam{cam0}/data/{ts0}.png"
    )
    left_image = Image.open(filepath)
    left_image = left_image.convert("RGB")


def track_point(x0: float, y0: float):
    global curr_reader, curr_flow, curr_certs

    # ipdb.set_trace()

    x1, y1, certainty = curr_reader.get_target_keypoint(curr_flow, curr_certs, x0, y0)

    # certainty = 1 if certainty > 0.6 else 0

    return x1, y1, certainty
