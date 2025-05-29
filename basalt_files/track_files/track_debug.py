import sys

print(f"{sys.path=}")

try:
    print(">>> Importing")

    import ipdb
    import numpy as np
    from PIL import Image
    import torch

    from track_utils import (
        clear_and_make_directory,
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
    "MOO12_freemovement_long_session": -1686080,
}

DATASET = "MOO12_freemovement_long_session"
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


import csv

track_debug_path = f"D:/thesis_code/track_debug/{DATASET}/"
clear_and_make_directory(f"{track_debug_path}/cam0")
clear_and_make_directory(f"{track_debug_path}/cam1")


def save_keypoints_csv(filepath, incoming_missed_kps):
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kpid", "x", "y", "x_guess", "y_guess"])
        for kpid, x, y, x_guess, y_guess in incoming_missed_kps:
            writer.writerow([kpid, x, y, x_guess, y_guess])


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

        if cam0 not in [0, 1]:
            return []

        guesses = dict(zip(kpids_guesses, keypoints_guesses))

        global cam
        cam = cam0

        global prev_ts, curr_ts
        ts0 -= TIME_OFFSET
        ts1 -= TIME_OFFSET
        prev_ts = ts0
        curr_ts = ts1

        # Read Images

        global left_image
        filepath = f"E:/thesis_code/datasets/monado_slam/{DATASET}/mav0/cam{cam0}/data/{ts0}.png"
        left_image = Image.open(filepath)
        left_image = left_image.convert("RGB")

        global right_image
        filepath = f"E:/thesis_code/datasets/monado_slam/{DATASET}/mav0/cam{cam0}/data/{ts1}.png"
        right_image = Image.open(filepath)
        right_image = right_image.convert("RGB")

        incoming_missed_kps_path = (
            f"{track_debug_path}/cam{cam0}/{ts0}_incoming_missed_kps.csv"
        )

        incoming_missed_kps = []

        for kpid_A, kp_A, lifetime in zip(kpids_A, keypoints_A, lifetimes_A):
            if kpid_A in kpids_B:
                continue

            if lifetime < 3:
                continue

            x, y = kp_A
            x_guess, y_guess = guesses[kpid_A]
            incoming_missed_kps.append((kpid_A, x, y, x_guess, y_guess))

        save_keypoints_csv(incoming_missed_kps_path, incoming_missed_kps)

        return []

    except Exception as e:
        print(e)
        ipdb.set_trace()
