import sys

print(f"{sys.path=}")

try:
    print(">>> Importing")

    import ipdb
    import numpy as np
    from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter

    from skimage import feature

    # import cv2
    # import albumentations as A
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms as T
    import lightning.pytorch as pl

    from esda.moran import Moran
    from libpysal.weights import lat2W

    from model_A import (
        MatcherModel,
        checkpoint_path,
        patch_normalize,
    )

    from track_utils import (
        crop_image_alb,
        cut_patches,
        min_max_normalize,
    )

    print(">>> Imported")
except Exception as e:
    print(">>> Failed to import", e)


TIME_OFFSETS = {
    # MG
    "MGO07_mapping_easy": 191231,
    "MGO08_mapping_hard": 261838,
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

DATASET = "MOO10_short_2_panorama"
TIME_OFFSET = TIME_OFFSETS[DATASET]
DATASET_PATH = f"D:/thesis_code/datasets/output/output_all/basalt/monado_slam/{DATASET}"

CONFIDENCE_THRESHOLD = 0.98


torch.set_float32_matmul_precision("medium")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")


cam = None

prev_ts = None
curr_ts = None

left_image = None
right_image = None
image_bw = None


class Light(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = MatcherModel()
        self.model = self.model.to(device)

    def forward(self, reference_patches, target_patches, reference_coords):
        pred = self.model(
            reference_patches,
            target_patches,
            reference_coords,
        )

        return pred


try:
    light = Light.load_from_checkpoint(checkpoint_path)
    print("Checkpoint Loaded")
except Exception as e:
    print("Error loading checkpoint")
    print(e)


def moran_skip(crop1, threshold=0.7):
    crop1_np = np.array(crop1)

    # Convert image to 1D array
    flattened_patch = crop1_np.ravel()

    # Create a spatial weight matrix
    w = lat2W(*crop1_np.shape)
    moran = Moran(flattened_patch, w)

    must_skip = abs(moran.I) < threshold

    return must_skip, f"{moran.I:.2f}"


def edge_skip(crop1, min_edge_density_threshold=0.012):
    p = crop1.copy().convert("L")
    # p = p.filter(ImageFilter.MedianFilter(size=3))

    patch_array = np.array(p)

    sigma = 1
    edges = (
        feature.canny(
            patch_array,
            sigma=sigma,
        ).astype(np.uint8)
        * 255
    )

    # Count non-zero pixels (edges)
    num_edges = np.count_nonzero(edges)
    total_pixels = patch_array.shape[0] * patch_array.shape[1]
    edge_density = round(num_edges / total_pixels, 4)

    p = Image.fromarray(edges).convert("RGB")

    must_skip = edge_density < min_edge_density_threshold

    return must_skip, f"{edge_density}", p


def skip(kpids_A_not_in_B, keypoints_A_not_in_B):
    moran_patch_size = 16
    reference_patches_pil_for_moran, reference_patches_coords_moran = cut_patches(
        image_bw, keypoints_A_not_in_B, box_size=moran_patch_size
    )

    moran_threshold = 0.66
    moran_must_skips = [
        moran_skip(crop, threshold=moran_threshold)[0]
        for crop in reference_patches_pil_for_moran
    ]

    box_size = 82  # 32  82

    reference_patches_pil_for_edge, reference_patches_coords_for_edge = cut_patches(
        image_bw, keypoints_A_not_in_B, box_size=box_size
    )

    min_edge_density_threshold = 0.02

    edge_must_skips = [
        edge_skip(
            crop,
            min_edge_density_threshold=min_edge_density_threshold,
        )[0]
        for crop in reference_patches_pil_for_edge
    ]

    # moran_must_skips = edge_must_skips

    pipeline_result_must_skips = [
        bool(s1 or s2) for s1, s2 in zip(moran_must_skips, edge_must_skips)
    ]

    kpids_A_not_in_B_to_be_used = [
        k
        for k, must_skip in zip(kpids_A_not_in_B, pipeline_result_must_skips)
        if not must_skip
    ]

    keypoints_A_not_in_B_to_be_used = [
        k
        for k, must_skip in zip(keypoints_A_not_in_B, pipeline_result_must_skips)
        if not must_skip
    ]

    return kpids_A_not_in_B_to_be_used, keypoints_A_not_in_B_to_be_used


def track_pair(
    cam0: int,
    ts0: int,
    cam1: int,
    ts1: int,
    kpids_A,
    keypoints_A,
    kpids_B,
    keypoints_B,
    kpids_guesses,
    keypoints_guesses,
    lifetimes_A,
):
    try:
        assert cam0 == cam1, "The dataset does not provide inter-camera matches"
        assert set(kpids_A) == set(
            kpids_guesses
        ), "The keypoints in A and guesses must be same"

        guesses = dict(zip(kpids_guesses, keypoints_guesses))

        global cam
        cam = cam0

        global prev_ts, curr_ts
        ts0 -= TIME_OFFSET
        ts1 -= TIME_OFFSET
        prev_ts = ts0
        curr_ts = ts1

        # Read Images

        global left_image, image_bw
        filepath = f"D:/thesis_code/datasets/monado_slam/{DATASET}/mav0/cam{cam0}/data/{ts0}.png"
        left_image = Image.open(filepath)
        left_image = left_image.convert("RGB")

        image_bw = left_image.convert("L")

        global right_image
        filepath = f"D:/thesis_code/datasets/monado_slam/{DATASET}/mav0/cam{cam0}/data/{ts1}.png"
        right_image = Image.open(filepath)
        right_image = right_image.convert("RGB")

        # Get Missed Points

        kpids_A_not_in_B, keypoints_A_not_in_B = [], []

        # for kpid_A, kp_A in zip(kpids_A, keypoints_A):
        for kpid_A, kp_A, lifetime in zip(kpids_A, keypoints_A, lifetimes_A):
            if lifetime < 3:
                continue

            if kpid_A not in kpids_B:
                kpids_A_not_in_B.append(kpid_A)
                keypoints_A_not_in_B.append(kp_A)

        # kpids_A_not_in_B_to_be_used, keypoints_A_not_in_B_to_be_used = skip(
        #     kpids_A_not_in_B, keypoints_A_not_in_B
        # )

        kpids_A_not_in_B_to_be_used, keypoints_A_not_in_B_to_be_used = (
            kpids_A_not_in_B,
            keypoints_A_not_in_B,
        )

        reference_patches, target_patches, reference_coords = [], [], []
        adjustments = []

        for kpid_A, kp_A in zip(
            kpids_A_not_in_B_to_be_used, keypoints_A_not_in_B_to_be_used
        ):
            x, y = kp_A

            left_crop, left_crop_keypoint, left, upper = crop_image_alb(
                left_image, [x, y], patch_size=32
            )

            left_crop = patch_normalize(left_crop)
            left_crop = min_max_normalize(left_crop, min_val=0.0, max_val=1.0)
            reference_patches.append(left_crop)

            reference_coords.append(left_crop_keypoint)

            x, y = guesses[kpid_A]

            right_crop, _, left, upper = crop_image_alb(
                right_image, [x, y], patch_size=32
            )

            adjustments.append((left, upper))

            right_crop = patch_normalize(right_crop)
            right_crop = min_max_normalize(right_crop, min_val=0.0, max_val=1.0)
            target_patches.append(right_crop)

        if len(reference_coords) <= 1:
            return []

        reference_patches = torch.stack(reference_patches).to(device)
        target_patches = torch.stack(target_patches).to(device)
        reference_coords = torch.tensor(reference_coords, dtype=torch.float32).to(
            device
        )

        # call light

        pred = light(reference_patches, target_patches, reference_coords)
        coords, confidences = pred

        # process confidences

        mask = confidences.squeeze() >= CONFIDENCE_THRESHOLD

        filtered_coords = coords[mask]
        filtered_adjustments = [adj for adj, keep in zip(adjustments, mask) if keep]
        filtered_kpids = [
            kp for kp, keep in zip(kpids_A_not_in_B_to_be_used, mask) if keep
        ]

        # prepare result

        # ipdb.set_trace()

        result = []

        for kpid, (x, y), (adj_x, adj_y) in zip(
            filtered_kpids, filtered_coords, filtered_adjustments
        ):
            x, y = x.detach().cpu().item(), y.detach().cpu().item()
            x, y = (x + 1) * 15.5, (y + 1) * 15.5

            # convert patch coords to image coords
            x_im, y_im = x + adj_x, y + adj_y

            # ipdb.set_trace()

            assert (
                x_im < 640 and y_im < 480
            ), "Predicted image coords are beyond the limits"

            result.append((kpid, x_im, y_im))

        # ipdb.set_trace()

        print(f"{len(result)=}")

        return result

    except Exception as e:
        print(e)
        ipdb.set_trace()
