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

    # from model_smith import (
    #     MatcherModel,
    #     checkpoint_path,
    #     patch_normalize,
    # )

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

DATASET = "MOO09_short_1_updown"
TIME_OFFSET = TIME_OFFSETS[DATASET]
DATASET_PATH = f"E:/thesis_code/datasets/output/output_all/basalt/monado_slam/{DATASET}"

torch.set_float32_matmul_precision("medium")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")


cam = None

prev_ts = None
curr_ts = None

left_image = None
right_image = None
image_bw = None

counts = 0
frames_processed = 0


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

        global cam, frames_processed
        cam = cam0

        if cam == 0:
            frames_processed += 1

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

        # Get Missed Points

        kpids_A_not_in_B, keypoints_A_not_in_B = [], []

        for kpid_A, kp_A, lifetime in zip(kpids_A, keypoints_A, lifetimes_A):
            if lifetime < 3:
                continue

            if kpid_A not in kpids_B:
                kpids_A_not_in_B.append(kpid_A)
                keypoints_A_not_in_B.append(kp_A)

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

        # Prepare Patches & Points

        used_ref_keypoints, used_est_keypoints = [], []
        used_kpids = []

        for kpid_A, kp_A in zip(
            kpids_A_not_in_B_to_be_used, keypoints_A_not_in_B_to_be_used
        ):
            used_kpids.append(kpid_A)
            used_ref_keypoints.append((x, y))
            x, y = guesses[kpid_A]
            used_est_keypoints.append((x, y))

        if len(used_est_keypoints) <= 1:
            return []

        # call light
        target_coords = used_est_keypoints

        # Prepare Result

        result = []

        for (
            kpid,
            (target_x, target_y),
            (ref_x, ref_y),
            (est_x, est_y),
        ) in zip(
            used_kpids,
            target_coords,
            used_ref_keypoints,
            used_est_keypoints,
        ):
            if target_x < 640 and target_y < 480:
                result.append((kpid, target_x, target_y))
            else:
                print(
                    f"Predicted image coords are beyond the limits : {target_x=}, {target_y=}"
                )

            # Target Prediction

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

        if cam == 0:
            global counts
            counts += len(result)
            avg_val = counts / frames_processed
            print(f"=> {counts, avg_val, len(result)=}")
            print(f"=> {frames_processed=}")

            if len(result) > 0:
                concat_image = Image.new(
                    "RGB",
                    (
                        left_image_pil.width + right_image_pil.width,
                        max(left_image_pil.height, right_image_pil.height),
                    ),
                )

                concat_image.paste(left_image_pil, (0, 0))
                concat_image.paste(right_image_pil, (left_image_pil.width, 0))

                concat_image.save(f"./debug_images/{frames_processed}_{ts0}_{ts1}.png")

        return result

    except Exception as e:
        print(e)
        ipdb.set_trace()
