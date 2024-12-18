import json

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from config import config
from utils import logger


def to_transformation_matrix(d):
    px, py, pz = d['px'], d['py'], d['pz']
    qx, qy, qz, qw = d['qx'], d['qy'], d['qz'], d['qw']

    rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

    # 4x4 transformation matrix
    transformation = np.eye(4)  # Identity matrix
    transformation[:3, :3] = rotation  # Set rotation
    transformation[:3, 3] = [px, py, pz]  # Set translation

    return transformation


def read_calib_json():
    calib_path = config.paths.basalt.calib_json
    # calib_path = r"D:/thesis_code/datasets/monado_slam/M_monado_datasets_MO_odyssey_plus_extras_calibration.json"
    with open(calib_path, 'r') as file:
        data = json.load(file)
        data = data['value0']['T_imu_cam']
        return data


def read_imu_csv():
    imu_path = config.paths.basalt.imu_csv
    imu = pd.read_csv(imu_path, header=0, names=('timestamp', 'w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z'))
    imu['ts'] = pd.to_datetime(imu['timestamp'], unit='ns')
    imu = imu.sort_values(by='ts').reset_index(drop=True)

    return imu


def read_gt_csv():
    gt_path = config.paths.basalt.gt_csv
    # gt_path = r'D:\thesis_code\datasets\monado_slam\MOO07_mapping_easy\mav0\gt\data.csv'
    gt = pd.read_csv(gt_path, header=0, names=('timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz'))
    gt['ts'] = pd.to_datetime(gt['timestamp'], unit='ns')
    gt = gt.sort_values(by='ts').reset_index(drop=True)

    return gt


def read_images_csv():
    files_path = config.paths.basalt.images_csv
    # files_path = r"D:\thesis_code\datasets\monado_slam\MOO07_mapping_easy\mav0\cam0\data.csv"
    files = pd.read_csv(files_path, header=0, names=('timestamp', 'filename'))
    files['ts'] = pd.to_datetime(files['timestamp'], unit='ns')
    files = files.sort_values(by='ts').reset_index(drop=True)
    logger.info(f"files.shape : {files.shape}")

    def image_exists(filename):
        image_path = Path(f"{config.paths[config.task.name].images}/{filename.strip()}")
        return image_path.exists()

    filtered_files = files[files['filename'].apply(image_exists)].reset_index(drop=True)
    logger.info(f"filtered_files.shape : {filtered_files.shape}")

    return filtered_files


def align_rows(files, gt):
    aligned_df = pd.merge_asof(
        files, gt,
        on="timestamp",
        direction="backward",
    ).dropna()

    aligned_df.to_csv(config.paths.basalt.aligned_csv, index=False, header=True)
    logger.info(f"aligned_df.shape : {aligned_df.shape}")

    return aligned_df


def filter_rows(aligned_df, T_i_c0):
    displacement_threshold = 0.02  # meters - 2 cm
    angle_threshold = 20 # 10  # degrees

    T_w_i = None
    T_w_c_t1 = None

    keyframes_indices = []

    for index, entry in aligned_df.iterrows():
        if T_w_i is None:
            T_w_i = to_transformation_matrix(entry)
            T_w_c_t1 = T_w_i @ T_i_c0
            keyframes_indices.append(index)
            continue

        T_w_i = to_transformation_matrix(entry)
        T_w_c_t2 = T_w_i @ T_i_c0

        R_t1 = T_w_c_t1[:3, :3]
        R_t2 = T_w_c_t2[:3, :3]

        R_relative = np.linalg.inv(R_t1) @ R_t2
        euler_angles = R.from_matrix(R_relative).as_euler('xyz', degrees=True)
        angle_norm = np.linalg.norm(euler_angles)

        if angle_norm > angle_threshold:
            keyframes_indices.append(index)
            T_w_c_t1 = T_w_c_t2
            continue

        location_t1 = T_w_c_t1[:3, 3]
        location_t2 = T_w_c_t2[:3, 3]
        displacement = np.linalg.norm(location_t2 - location_t1)

        if displacement > displacement_threshold:
            keyframes_indices.append(index)
            T_w_c_t1 = T_w_c_t2

    keyframes_df = aligned_df.iloc[keyframes_indices].reset_index(drop=True)
    keyframes_df = keyframes_df.drop(columns=['ts_x', 'ts_y'])
    keyframes_df.to_csv(config.paths.basalt.keyframes_csv, index=False, header=True)

    # files_path = r"D:\thesis_code\datasets\monado_slam\MOO07_mapping_easy\mav0\cam0\data_filtered.csv"
    # keyframes_df.to_csv(files_path, index=False, header=True)

    logger.info(f"keyframes_df.shape : {keyframes_df.shape}")


def align_and_filter_rows():
    for track in config.task.tracks:
        config.task.track = track

        logger.info(f'Filter {track}')
        config.task.dataset_kind = track[:2]

        gt = read_gt_csv()

        calibs = read_calib_json()

        for index, calib in enumerate(calibs):
            if index == 2:
                break

            config.task.cam = f'cam{index}'
            logger.info(f'cam{index}')

            T_i_c = to_transformation_matrix(calib)

            filenames = read_images_csv()
            aligned_df = align_rows(filenames, gt)
            filter_rows(aligned_df, T_i_c)


def main():
    align_and_filter_rows()


if __name__ == '__main__':
    main()
