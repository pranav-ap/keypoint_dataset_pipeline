import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
import seaborn as sns
sns.set_theme(style="darkgrid")


def read_image_and_size(image_path):
    image = Image.open(image_path)
    W, H = image.size
    return image, W, H


def pillow_to_cv2(pil_image):
    # Convert Pillow image to NumPy array (RGB format)
    numpy_image = np.array(pil_image)
    # Convert RGB to BGR format (OpenCV uses BGR)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image


def plot_single(a):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax[0].imshow(a)
    ax[0].grid(False)  # turn off grid
    plt.show()


def plot_pair(a, b):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(a)
    ax[0].grid(False)
    ax[1].imshow(b)
    ax[1].grid(False)
    plt.show()


def draw_keypoints(image, keypoints):
    keypoints = [cv2.KeyPoint(x, y, 1.) for x, y in keypoints.cpu().numpy()]
    image = np.array(image)

    ret = cv2.drawKeypoints(
        image,
        keypoints,
        None
    )

    return ret


def draw_matches(image_A, keypoints_A, image_B, keypoints_B):
    keypoints_A = [cv2.KeyPoint(x, y, 1.) for x, y in keypoints_A.cpu().numpy()]
    keypoints_B = [cv2.KeyPoint(x, y, 1.) for x, y in keypoints_B.cpu().numpy()]

    count = min(10, len(keypoints_A))
    matches_A_to_B = [cv2.DMatch(idx, idx, 0.) for idx in range(count)]

    image_A, image_B = np.array(image_A), np.array(image_B)

    match_image = cv2.drawMatches(
        image_A, keypoints_A,
        image_B, keypoints_B,
        matches_A_to_B,
        outImg=None,
        matchesThickness=3,
    )

    return match_image


def tensor_to_pil_image(tensor):
    tensor = tensor.clone().detach()  # Detach and clone to create a safe copy

    # If the tensor is 4D (batch), remove the batch dimension
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    # Convert the tensor to a format PIL can handle
    if tensor.ndim == 3 and tensor.shape[0] == 1:  # Grayscale image
        tensor = tensor.squeeze(0)  # Remove channel dimension for grayscale

    # Scale the values to [0, 255] for uint8 images
    tensor = tensor.clone().detach()  # Detach from computation graph
    if tensor.min() < 0 or tensor.max() > 1:  # Normalize to [0, 1] if needed
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    tensor = (tensor * 255).byte()  # Convert to uint8

    # Convert to PIL image
    return T.ToPILImage()(tensor)
