import sys

print(f"{sys.path=}")

try:
    print(">>> Importing")

    import ipdb
    import os
    import numpy as np
    from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter

    from skimage import feature

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms as T
    import lightning.pytorch as pl

    from model_ryan import (
        MatcherModel,
        checkpoint_path,
        patch_normalize,
    )

    from track_utils import (
        crop_image_alb,
        cut_patches,
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

DATASET = "MGO10_short_2_panorama"
TIME_OFFSET = TIME_OFFSETS[DATASET]
DATASET_PATH = f"E:/thesis_code/datasets/output/output_all/basalt/monado_slam/{DATASET}"

# True  False

USE_SKIP = False
STORE_DEBUG_IMAGES = False

USE_LIFETIME_FOR_FILTERING = True
USE_MODEL_CONFIDENCE_FOR_FILTERING = False
CONFIDENCE_THRESHOLD = 0.98  # 0.9 0.98

USE_ESTIMATES_AS_PREDICTIONS = False

torch.set_float32_matmul_precision("medium")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")


cam = None

prev_ts = None
curr_ts = None

left_image = None
right_image = None
image_bw = None

frames_processed = 0
frames_processed_cam = 0
num_keypoints_without_matches = 0
num_keypoints_after_LT_filtering = 0
num_keypoints_after_MODEL_filtering = 0


class Light(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = MatcherModel().to(device)

    def forward(self, reference_patches, target_patches, estimates):
        coords_delta_norm_pred, confs_pred = self.model(
            reference_patches,
            target_patches,
            estimates,
        )

        coords_delta_pred = coords_delta_norm_pred.float() * ((32 - 1) / 2)
        coords_pred = estimates + coords_delta_pred
        coords_pred = torch.clamp(coords_pred, min=0.0, max=float(32))

        return coords_pred, confs_pred


try:
    print("Loading checkpoint")
    light = Light.load_from_checkpoint(checkpoint_path)
    # light.eval()
    print("Checkpoint Loaded")
except Exception as e:
    print("Error loading checkpoint")
    print(e)


if not os.path.exists("./debug_images"):
    print("Creating debug_images directory")
    os.makedirs("./debug_images")
else:
    print("debug_images directory already exists")
    print("Cleaning debug_images directory")
    for file in os.listdir("./debug_images"):
        file_path = os.path.join("./debug_images", file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(e)


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
    box_size = 32

    reference_patches_pil_for_edge, reference_patches_coords_for_edge = cut_patches(
        image_bw, keypoints_A_not_in_B, box_size=box_size
    )

    min_edge_density_threshold = 0.1  #  0.08

    edge_must_skips = [
        edge_skip(
            crop,
            min_edge_density_threshold=min_edge_density_threshold,
        )[0]
        for crop in reference_patches_pil_for_edge
    ]

    pipeline_result_must_skips = [bool(s1) for s1 in edge_must_skips]

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

        global num_keypoints_without_matches, num_keypoints_after_LT_filtering
        global num_keypoints_after_MODEL_filtering

        global cam, frames_processed, frames_processed_cam
        cam = cam0

        if cam not in [0, 1]:
            return []

        frames_processed += 1
        if cam == 0:
            frames_processed_cam += 1

        global prev_ts, curr_ts
        ts0 -= TIME_OFFSET
        ts1 -= TIME_OFFSET
        prev_ts = ts0
        curr_ts = ts1

        # Read Images

        global left_image, image_bw
        filepath = f"E:/thesis_code/datasets/monado_slam/{DATASET}/mav0/cam{cam0}/data/{ts0}.png"
        left_image = Image.open(filepath)
        left_image = left_image.convert("RGB")

        image_bw = left_image.convert("L")

        global right_image
        filepath = f"E:/thesis_code/datasets/monado_slam/{DATASET}/mav0/cam{cam0}/data/{ts1}.png"
        right_image = Image.open(filepath)
        right_image = right_image.convert("RGB")

        left_image_pil = left_image.copy()
        draw_left = ImageDraw.Draw(left_image_pil)

        right_image_pil = right_image.copy()
        draw_right = ImageDraw.Draw(right_image_pil)

        # Get Unmissed Missed Points

        kpids_A_not_in_B, keypoints_A_not_in_B = [], []

        for kpid_A, kp_A, lifetime in zip(kpids_A, keypoints_A, lifetimes_A):
            if kpid_A not in kpids_B:
                num_keypoints_without_matches += 1

                if USE_LIFETIME_FOR_FILTERING and lifetime < 3:
                    continue

                num_keypoints_after_LT_filtering += 1

                kpids_A_not_in_B.append(kpid_A)
                keypoints_A_not_in_B.append(kp_A)

        if STORE_DEBUG_IMAGES:
            for kpid_A, kp_A in zip(kpids_A_not_in_B, keypoints_A_not_in_B):
                x, y = kp_A
                radius = 3

                # Reference

                draw_left.ellipse(
                    (
                        int(x) - radius,
                        int(y) - radius,
                        int(x) + radius,
                        int(y) + radius,
                    ),
                    outline="lime",
                    width=2,
                )

                draw_left.text(
                    (int(x) + 6, int(y) - 6),
                    str(kpid_A),
                    fill="lime",
                )

                x, y = guesses[kpid_A]

                draw_right.ellipse(
                    (
                        int(x) - radius,
                        int(y) - radius,
                        int(x) + radius,
                        int(y) + radius,
                    ),
                    outline="red",
                    width=2,
                )

                draw_right.text((int(x) + 6, int(y) - 6), str(kpid_A), fill="red")

        kpids_A_not_in_B_to_be_used, keypoints_A_not_in_B_to_be_used = (
            kpids_A_not_in_B,
            keypoints_A_not_in_B,
        )

        if USE_SKIP:
            kpids_A_not_in_B_to_be_used, keypoints_A_not_in_B_to_be_used = skip(
                kpids_A_not_in_B, keypoints_A_not_in_B
            )

        # Prepare Patches & Points

        reference_patches, target_patches, estimates = [], [], []
        adjustments = []
        used_ref_keypoints, used_est_keypoints = [], []
        used_kpids = []

        for kpid_A, kp_A in zip(
            kpids_A_not_in_B_to_be_used, keypoints_A_not_in_B_to_be_used
        ):
            x, y = kp_A

            left_crop, left_crop_keypoint, left, upper = crop_image_alb(
                left_image, [x, y], patch_size=32
            )

            if left_crop_keypoint[0] != 16 or left_crop_keypoint[1] != 16:
                continue

            used_kpids.append(kpid_A)

            left_crop = left_crop.convert("L")
            left_crop = patch_normalize(left_crop)
            reference_patches.append(left_crop)

            used_ref_keypoints.append((x, y))

            x, y = guesses[kpid_A]
            used_est_keypoints.append((x, y))

            right_crop, right_crop_keypoint, left, upper = crop_image_alb(
                right_image, [x, y], patch_size=32
            )

            right_crop = right_crop.convert("L")
            right_crop = patch_normalize(right_crop)
            target_patches.append(right_crop)

            estimates.append(right_crop_keypoint)
            adjustments.append((left, upper))

        if len(estimates) <= 1:
            return []

        reference_patches = torch.stack(reference_patches).to(device)
        target_patches = torch.stack(target_patches).to(device)
        used_ref_keypoints = torch.tensor(used_ref_keypoints, dtype=torch.float32).to(
            device
        )
        used_est_keypoints = torch.tensor(used_est_keypoints, dtype=torch.float32).to(
            device
        )
        estimates = torch.tensor(estimates, dtype=torch.float32).to(device)

        # call light

        batch_size = 32

        all_target_coords = []
        all_confidences = []

        for i in range(0, len(reference_patches), batch_size):
            ref_batch = reference_patches[i : i + batch_size]
            tgt_batch = target_patches[i : i + batch_size]
            est_batch = estimates[i : i + batch_size]

            coords, confs = light(ref_batch, tgt_batch, est_batch)
            all_target_coords.append(coords)
            all_confidences.append(confs)

        target_coords = torch.cat(all_target_coords, dim=0)
        confidences = torch.cat(all_confidences, dim=0)

        # target_coords, confidences = light(reference_patches, target_patches, estimates)

        if USE_ESTIMATES_AS_PREDICTIONS:
            target_coords = estimates

        # Prepare Result

        result = []

        for (
            kpid,
            (target_x, target_y),
            (adj_x, adj_y),
            (ref_x, ref_y),
            (est_x, est_y),
            conf,
        ) in zip(
            used_kpids,
            target_coords,
            adjustments,
            used_ref_keypoints,
            used_est_keypoints,
            confidences,
        ):
            if USE_MODEL_CONFIDENCE_FOR_FILTERING and conf < CONFIDENCE_THRESHOLD:
                continue

            num_keypoints_after_MODEL_filtering += 1

            target_x, target_y = (
                target_x.detach().cpu().item(),
                target_y.detach().cpu().item(),
            )

            # convert patch coords to image coords

            target_x, target_y = target_x + adj_x, target_y + adj_y

            if target_x < 640 and target_y < 480:
                result.append((kpid, target_x, target_y))
            else:
                print(
                    f"Predicted image coords are beyond the limits : {target_x=}, {target_y=}"
                )

            # Target Prediction

            if STORE_DEBUG_IMAGES:
                radius = 1

                draw_right.ellipse(
                    (
                        int(target_x) - radius,
                        int(target_y) - radius,
                        int(target_x) + radius,
                        int(target_y) + radius,
                    ),
                    outline="lime",
                    width=2,
                )

                draw_right.text(
                    (int(target_x) + 6, int(target_y) - 6),
                    str(kpid),
                    fill="lime",
                )

        if cam == 0 and STORE_DEBUG_IMAGES and len(result) > 0:
            concat_image = Image.new(
                "RGB",
                (
                    left_image_pil.width + right_image_pil.width,
                    max(left_image_pil.height, right_image_pil.height),
                ),
            )

            concat_image.paste(left_image_pil, (0, 0))
            concat_image.paste(right_image_pil, (left_image_pil.width, 0))

            concat_image.save(f"./debug_images/{frames_processed_cam}_{ts0}_{ts1}.png")

        print(f"{len(result)=}")

        print(f"Number of Frames Processed: {frames_processed}")

        print(f"Number of keypoints without matches: {num_keypoints_without_matches}")
        print(
            f"Number of keypoints after LT filtering: {num_keypoints_after_LT_filtering}"
        )
        print(
            f"Number of keypoints after MODEL filtering: {num_keypoints_after_MODEL_filtering}"
        )

        print(
            f"Average Number of Keypoints added per frame: {num_keypoints_after_MODEL_filtering / frames_processed}"
        )

        print(
            f"Percent of Keypoints used : {100 * num_keypoints_after_MODEL_filtering / num_keypoints_without_matches}"
        )

        print("---------------")

        return result

    except Exception as e:
        print(e)
        ipdb.set_trace()
