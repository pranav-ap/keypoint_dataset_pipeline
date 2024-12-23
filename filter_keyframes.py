import json
import shutil

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from config import config
from utils import logger, make_clear_directory


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

    T_list = data['value0']['T_imu_cam']
    intrinsics_list = data['value0']['intrinsics']
    return T_list, intrinsics_list


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
    logger.info(f"Original files.shape : {files.shape}")

    def image_exists(filename):
        image_path = Path(f"{config.paths[config.task.name].images}/{filename.strip()}")
        return image_path.exists()

    filtered_files = files[files['filename'].apply(image_exists)].reset_index(drop=True)
    if filtered_files.shape[0] < files.shape[0]:
        logger.debug(f"After Image Exists Check files.shape : {filtered_files.shape}")

    return filtered_files


def align_rows(files, gt):
    aligned_df = pd.merge_asof(
        files, gt,
        on="timestamp",
        direction="backward",
    ).dropna()

    # aligned_df.to_csv(config.paths.basalt.aligned_csv, index=False, header=True)
    # logger.debug(f"aligned_df.shape : {aligned_df.shape}")

    return aligned_df


def filter_rows(aligned_df, T_i_c, intrinsics):
    # csv in meters
    # 20 cm to 0.2 m
    displacement_threshold =config.task.displacement_threshold / 100  
    # csv in radians
    # 20 degrees to 0.34 rad
    angle_threshold = config.task.angle_threshold * np.pi / 180  

    T_w_i = None
    T_w_c_t1 = None

    keyframes_indices = []

    for index, entry in aligned_df.iterrows():
        if T_w_i is None:
            T_w_i = to_transformation_matrix(entry)
            T_w_c_t1 = T_w_i @ T_i_c
            keyframes_indices.append(index)
            continue

        T_w_i = to_transformation_matrix(entry)
        T_w_c_t2 = T_w_i @ T_i_c

        R_t1 = T_w_c_t1[:3, :3]
        R_t2 = T_w_c_t2[:3, :3]

        R_relative = np.linalg.inv(R_t1) @ R_t2
        axis_angles = R.from_matrix(R_relative).as_rotvec()
        angle_norm = np.linalg.norm(axis_angles)

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

    if config.task.output_folder_name == 'output_filtered_test':
        image_names = keyframes_df['filename'].tolist()

        for name in image_names:
            image_path_from = Path(f"{config.paths[config.task.name].images}/{name.strip()}")
            image_path_to = Path(f"{config.paths[config.task.name].output_cam}/images/{name.strip()}")
            shutil.copy(image_path_from, image_path_to)

    logger.info(f"keyframes_df.shape : {keyframes_df.shape}")


def filter_blurred_rows(aligned_df, T_i_c, intrinsics):
    blur_threshold = config.task.blur_threshold  

    keyframes_indices = []
    
    for i in range(len(aligned_df) - 1):
        row1, row2 = aligned_df.iloc[i], aligned_df.iloc[i + 1]
        
        T_w_i = to_transformation_matrix(row1)
        T_w_c_t1 = T_w_i @ T_i_c

        T_w_i = to_transformation_matrix(row2)
        T_w_c_t2 = T_w_i @ T_i_c

        scaled_z = 2 * np.array([0, 0, 1, 1]) 
        V = np.linalg.inv(T_w_c_t2) @ T_w_c_t1 @ scaled_z 

        x, y, z = V[0], V[1], V[2]

        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        a = np.array([
            config.image.crop_image_shape[0] / 2, 
            config.image.crop_image_shape[1] / 2
        ])

        b = np.array([
            (fx * x / z) + cx, 
            (fy * y / z) + cy,
        ])

        result = np.linalg.norm(a - b)

        if result > blur_threshold:
            # print(result)
            if i not in keyframes_indices:
                keyframes_indices.append(i)
            if i + 1 not in keyframes_indices:
                keyframes_indices.append(i + 1)
     
    keyframes_df = aligned_df.iloc[keyframes_indices].reset_index(drop=True)
    keyframes_df = keyframes_df.drop(columns=['ts_x', 'ts_y'])
    keyframes_df.to_csv(config.paths.basalt.keyframes_csv, index=False, header=True)

    if config.task.output_folder_name == 'output_filtered_test':
        image_names = keyframes_df['filename'].tolist()

        for name in image_names:
            image_path_from = Path(f"{config.paths[config.task.name].images}/{name.strip()}")
            image_path_to = Path(f"{config.paths[config.task.name].output_cam}/images/{name.strip()}")
            shutil.copy(image_path_from, image_path_to)

    logger.info(f"keyframes_df.shape : {keyframes_df.shape}")


def align_and_filter_rows():
    for track in config.task.tracks:
        config.task.track = track

        logger.info(f'Filter {track}')
        config.task.dataset_kind = track[:2]

        gt = read_gt_csv()

        T_list, intrinsics_list = read_calib_json()

        for index, (T, intrinsics) in enumerate(zip(T_list, intrinsics_list)):
            if index == 2:
                break
                
            config.task.cam = f'cam{index}'
            logger.info(f'cam{index}')

            make_clear_directory(config.paths.basalt.output_cam)
            make_clear_directory(f'{config.paths.basalt.output_cam}/images')

            T_i_c = to_transformation_matrix(T)

            intrinsics = intrinsics['intrinsics']

            filenames = read_images_csv()
            aligned_df = align_rows(filenames, gt)

            if config.task.use_blur_filter:
                filter_blurred_rows(aligned_df, T_i_c, intrinsics)
            else:
                filter_rows(aligned_df, T_i_c, intrinsics)


def main():
    align_and_filter_rows()


if __name__ == '__main__':
    main()
