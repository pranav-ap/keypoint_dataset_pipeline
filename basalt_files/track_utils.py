try:
    print(">>> Importing in utils file")

    import ipdb
    import numpy as np
    import pandas as pd

    import math
    # import cv2
    import torch
    import torchvision.transforms as T
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    import os
    import shutil

    from pathlib import Path
    # from natsort import natsorted

    print(">>> Imported")
except Exception as e:
    print(">>> Failed to import", e)


def get_patch_boundary(image: Image.Image, center_point, patch_size):
    image_width, image_height = image.size
    x, y = center_point
    half_patch_size = patch_size // 2

    left, right = x - half_patch_size, x + half_patch_size
    upper, lower = y - half_patch_size, y + half_patch_size

    if left < 0:
        right += -left
        left = 0
    elif right > image_width:
        left -= right - image_width
        right = image_width

    if upper < 0:
        lower += -upper
        upper = 0
    elif lower > image_height:
        upper -= lower - image_height
        lower = image_height

    assert right > left
    assert right - left == patch_size
    assert lower > upper
    assert lower - upper == patch_size

    return left, upper, right, lower


def crop_image_alb(image: Image.Image, keypoint, patch_size=32):
    left, upper, right, lower = get_patch_boundary(image, keypoint, patch_size)

    patch = image.crop((left, upper, right, lower))
    assert patch.size[0] == patch.size[1]
    assert patch.size[0] == patch_size

    new_keypoint = keypoint[0] - left, keypoint[1] - upper

    return patch, new_keypoint, left, upper


def crop_image(image, x, y, patch_size=32):
    crop_width, crop_height = patch_size, patch_size
    center_x, center_y = x, y

    left = center_x - crop_width // 2
    top = center_y - crop_height // 2
    right = center_x + crop_width // 2
    bottom = center_y + crop_height // 2

    image = image.crop((left, top, right, bottom))

    return image


def get_stddev(patch):
    patch = patch.convert("L")
    img_patch = np.array(patch, dtype=np.float32)

    # Compute gradients in x and y directions
    gradient_x = np.diff(np.pad(img_patch, ((0, 0), (1, 0)), mode="reflect"), axis=1)
    gradient_y = np.diff(np.pad(img_patch, ((1, 0), (0, 0)), mode="reflect"), axis=0)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Calculate the standard deviation of the gradient magnitudes
    gradient_std = np.std(gradient_magnitude)

    # print(f"{gradient_std=}")

    # Check if the gradient variability is below the threshold
    return gradient_std  # < threshold


def get_stddev_im(image, x, y, patch_size=32):
    gray_image = image.convert("L")
    # gray_image = crop_image_alb(gray_image, [x, y], patch_size)
    gray_image = crop_image(gray_image, x, y, patch_size)

    img_array = np.array(gray_image, dtype=np.float32)

    # Compute gradients in x and y directions
    gradient_x = np.diff(np.pad(img_array, ((0, 0), (1, 0)), mode="reflect"), axis=1)
    gradient_y = np.diff(np.pad(img_array, ((1, 0), (0, 0)), mode="reflect"), axis=0)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Calculate the standard deviation of the gradient magnitudes
    gradient_std = np.std(gradient_magnitude)

    # print(f"{gradient_std=}")

    return gradient_std


############ VISUALIZE


to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

# denormalize = T.Compose(
#     [
#         T.Normalize(
#             mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#             std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
#         )
#     ]
# )

# denormalize = T.Compose([T.Normalize(mean=[-0.5 / 0.5], std=[1 / 0.5])])


def clear_and_make_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Delete the directory if it exists
    os.makedirs(directory)  # Recreate the empty directory


def min_max_normalize(tensor, min_val=0.0, max_val=1.0):
    """
    Perform Min-Max Normalization on a tensor.
    Args:
        tensor (torch.Tensor): Input tensor with pixel values.
        min_val (float): Minimum value for normalization (default: 0.0).
        max_val (float): Maximum value for normalization (default: 1.0).
    Returns:
        torch.Tensor: Min-Max normalized tensor.
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    # Scale the tensor to the desired range
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    normalized_tensor = normalized_tensor * (max_val - min_val) + min_val
    return normalized_tensor


def get_tensor_grid(pil_image):
    return to_tensor(pil_image).unsqueeze(0)


def show_batch(
    reference_patches,
    target_patches,
    patch_level_reference_coords,
    patch_level_target_coords,
    patch_level_target_coords_true,
    rotations_true=None,
    rotations=None,
    confidence_pred=None,
    limit_count=None,
    border_size=2,
    border_color="white",
    n_columns=2,
    just_gt=False,
    just_pred=False,
):
    assert limit_count is None or limit_count > 0

    num_patches = reference_patches.size(0)
    if limit_count is not None:
        num_patches = min(limit_count, num_patches)

    if rotations_true is not None:
        rotations_true = rotations_true.clone().detach()
        # Convert radians to degrees
        rotations_true = rotations_true * (180 / torch.pi)

    # print(rotations_true[:num_patches])

    if rotations is not None:
        rotations = rotations.clone().detach()

        if not just_gt:
            # Convert [-1, 1] to [-pi, pi]
            rotations = rotations * torch.pi

        # Convert radians to degrees
        rotations = rotations * (180 / torch.pi)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    patch_size = 128
    extra_col_gap = 0
    radius = 1

    gap_for_text = 25

    num_rows = (num_patches + n_columns - 1) // n_columns

    combined_width = (
        n_columns * (patch_size * 2 + border_size * 4) + (n_columns - 1) * extra_col_gap
    )
    combined_height = num_rows * (patch_size + border_size * 2 + gap_for_text)

    combined_image = Image.new(
        "RGB", (combined_width, combined_height), color=(255, 255, 255)
    )

    def prepare_patch(patch, x, y, color):
        # patch = denormalize(patch)
        patch = min_max_normalize(patch, min_val=0.0, max_val=1.0)
        patch = to_pil(patch)

        if patch.mode != "RGB":
            patch = patch.convert("RGB")

        draw_im = ImageDraw.Draw(patch)
        draw_im.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color)

        # height = 4
        # draw_im.line([(x, height + 10), (x, height)], fill="red", width=1)

        patch = patch.resize((patch_size, patch_size))
        patch = ImageOps.expand(patch, border=border_size, fill=border_color)
        return patch

    def prepare_target_patch(patch, x, y, a, b, rot=None):
        # patch = denormalize(patch)
        patch = min_max_normalize(patch, min_val=0.0, max_val=1.0)
        patch = to_pil(patch)

        if patch.mode != "RGB":
            patch = patch.convert("RGB")

        draw_im = ImageDraw.Draw(patch)

        if not just_gt:
            draw_im.ellipse(
                (x - radius, y - radius, x + radius, y + radius), outline="yellow"
            )

            # Compute the endpoints for the line
            # length = 4

            # radians = math.radians(float(rot))

            # x1 = x - length * math.sin(radians)
            # y1 = y + length * math.cos(radians)
            # x2 = x + length * math.sin(radians)
            # y2 = y - length * math.cos(radians)

            # draw_im.line([(x1, y1), (x2, y2)], fill="green", width=1)

        if not just_pred:
            draw_im.rectangle(
                (a - radius - 1, b - radius - 1, a + radius + 1, b + radius + 1),
                outline="green",
            )

        patch = patch.resize((patch_size, patch_size))
        patch = ImageOps.expand(patch, border=border_size, fill=border_color)
        return patch

    for i in range(num_patches):
        reference_patch = reference_patches[i]
        x, y = patch_level_reference_coords[i]
        reference_patch = prepare_patch(reference_patch, x, y, color="red")

        rotation_true = (
            f"{rotations_true[i].item():.2f}°" if rotations_true is not None else "-"
        )
        rotation = f"{rotations[i].item():.2f}°" if rotations is not None else "-"
        conf = (
            f"{confidence_pred[i].item():.2f}" if confidence_pred is not None else "-"
        )

        target_patch = target_patches[i]
        x, y = patch_level_target_coords[i]
        a, b = patch_level_target_coords_true[i]
        target_patch = prepare_target_patch(
            target_patch, x, y, a, b
        )  # , rotations[i].item())

        row = i // n_columns
        col = i % n_columns

        y = row * (patch_size + 2 * border_size + gap_for_text)
        x1 = col * (patch_size * 2 + border_size * 4) + col * extra_col_gap
        x2 = x1 + patch_size + 2 * border_size

        combined_image.paste(reference_patch, (x1, y))
        combined_image.paste(target_patch, (x2, y))

        # Draw rotation value below the patches
        draw = ImageDraw.Draw(combined_image)

        text_x = x1 + patch_size // 2
        text_y = y + patch_size + 2 * border_size + 10
        draw.text(
            (text_x, text_y), f"{rotation_true}", fill="black", anchor="mm", font=font
        )

        text_x = x2 - 5 + patch_size // 2
        text_y = y + patch_size + 2 * border_size + 10

        draw.text(
            (text_x, text_y),
            f"{rotation}, {conf}",
            fill="black",
            anchor="mm",
            font=font,
        )

    return combined_image


def show_image_grid(image_patches, N, patch_size):
    grid_size = N * patch_size

    grid_image = Image.new("RGB", (grid_size, grid_size), "white")

    for i, patch in enumerate(image_patches[: N * N]):
        row, col = divmod(i, N)
        grid_image.paste(patch, (col * patch_size, row * patch_size))

    return grid_image


def rearrange_patch_randomly(patch: np.ndarray) -> np.ndarray:
    # Flatten the patch to a 1D array
    flattened_patch = patch.flatten()

    # Randomly permute the flattened patch
    np.random.shuffle(flattened_patch)

    # Reshape it back to the original shape
    rearranged_patch = flattened_patch.reshape(patch.shape)

    return rearranged_patch


def find_common_elements(list1, list2):
    return list(set(list1) & set(list2))


def load_image(cam, ts, DATASET):
    filepatha = (
        f"D:/thesis_code/datasets/monado_slam/{DATASET}/mav0/cam{cam}/data/{ts}.png"
    )

    left_image = Image.open(filepatha)
    left_image = left_image.convert("RGB")

    return left_image


def load_missing_keypoints(cam, ts, DATASET):
    df = pd.read_csv(
        f"D:/thesis_code/track_debug/{DATASET}/cam{cam}/{ts}_incoming_missed_kps.csv",
        header=0, names=("kpid", "x", "y", "x_guess", "y_guess")
    )

    return df


def show_missed_keypoints(image, image_kpids, image_reference_coords):
    """
    >>> show_missed_keypoints(left_image, image_kpids, image_reference_coords)
    """

    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    for kpid, (x, y) in zip(image_kpids, image_reference_coords):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline="red")

        draw.text((x + 5, y - 5), str(kpid), fill="yellow")

    return image_copy


def filter_bad_reference_keypoints(
    image,
    image_kpids,
    image_reference_coords,
    skip=None,
    with_grid=False,
    moran_patch_size=8,
    box_size=64,
):
    patch_reference_coords = []
    reference_patches_pil = []
    reference_patches_pil_plain = {}
    chosen_kpids = []

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    image_bw = image.convert("L")

    for kpid, (x, y) in zip(image_kpids, image_reference_coords):
        crop_pil, crop_keypoint, _, _ = crop_image_alb(
            image_bw, [x, y], patch_size=moran_patch_size
        )

        p_x, p_y = crop_keypoint

        must_skip, mi = skip(crop_pil, p_x, p_y)

        outline = "red" if must_skip else "green"

        if not must_skip:
            chosen_kpids.append(kpid)

        patch_size = box_size
        crop_pil, crop_keypoint, _, _ = crop_image_alb(
            image, [x, y], patch_size=patch_size
        )

        p_x, p_y = crop_keypoint

        # draw

        reference_patches_pil_plain[kpid] = crop_pil.copy()

        crop_pil_copy = crop_pil.copy()
        draw = ImageDraw.Draw(crop_pil_copy)

        radius = 2
        draw.ellipse(
            (p_x - radius, p_y - radius, p_x + radius, p_y + radius),
            fill=outline,
            outline=outline,
        )

        disp_patch_size = box_size  # 64
        # crop_pil_copy = crop_pil_copy.resize((disp_patch_size, disp_patch_size))

        draw = ImageDraw.Draw(crop_pil_copy)
        draw.text((5, 5), str(kpid), fill="yellow", font=font)
        # draw.text((30, 45), mi, fill="orange", font=font)
        draw.text((box_size - 35, box_size - 25), mi, fill="orange", font=font)

        # store
        reference_patches_pil.append(crop_pil_copy)
        patch_reference_coords.append(crop_keypoint)

    if with_grid:
        grid = show_image_grid(reference_patches_pil, N=8, patch_size=disp_patch_size)
        # grid = grid.resize((512, 512))
        return grid, chosen_kpids, reference_patches_pil_plain, patch_reference_coords

    return chosen_kpids, reference_patches_pil_plain, patch_reference_coords


def filter_bad_reference_keypoints_prod(
    image,
    image_kpids,
    image_reference_coords,
    skip=None,
    moran_patch_size=8,
):
    chosen_kpids = []

    image_bw = image.convert("L")

    for kpid, (x, y) in zip(image_kpids, image_reference_coords):
        crop_pil, crop_keypoint, _, _ = crop_image_alb(
            image_bw, [x, y], patch_size=moran_patch_size
        )

        p_x, p_y = crop_keypoint

        must_skip, mi = skip(crop_pil, p_x, p_y)

        if not must_skip:
            chosen_kpids.append(kpid)

    return chosen_kpids


def cut_patches(
    image,
    image_coords,
    box_size=64,
):
    patches_pil = []
    patch_coords = []

    for x, y in image_coords:
        crop_pil, crop_keypoint, _, _ = crop_image_alb(
            image, [x, y], patch_size=box_size
        )

        patches_pil.append(crop_pil)
        patch_coords.append(crop_keypoint)

    return patches_pil, patch_coords


def annotate(kpids, patches_pil, patches_coords, must_skips, vals, box_size):
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    annotateed_patches_pil = []

    for kpid, crop_pil, (x, y), must_skip, val in zip(
        kpids, patches_pil, patches_coords, must_skips, vals
    ):
        crop_pil_copy = crop_pil.copy()
        draw = ImageDraw.Draw(crop_pil_copy)

        col = "red" if must_skip else "green"

        radius = 2
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=col,
            outline=col,
        )

        draw.text((5, 5), str(kpid), fill="yellow", font=font)
        draw.text((box_size - 55, box_size - 25), val, fill="orange", font=font)

        annotateed_patches_pil.append(crop_pil_copy)

    return annotateed_patches_pil


def create_patch_grid(patches, grid_size=None):
    """Creates a grid from a list of PIL image patches with an even number of columns."""
    if not patches:
        raise ValueError("Patch list is empty!")

    # Get patch dimensions
    patch_w, patch_h = patches[0].size
    num_patches = len(patches)

    # Determine grid size (rows, cols)
    if grid_size:
        grid_w, grid_h = grid_size
        if grid_w % 2 != 0:  # Ensure columns are even
            grid_w += 1
    else:
        grid_w = math.ceil(math.sqrt(num_patches))
        if grid_w % 2 != 0:  # Force even number of columns
            grid_w += 1
        grid_h = math.ceil(num_patches / grid_w)

    # Create blank canvas
    grid_img = Image.new("RGB", (grid_w * patch_w, grid_h * patch_h))

    # Paste patches into grid
    for idx, patch in enumerate(patches):
        x = (idx % grid_w) * patch_w
        y = (idx // grid_w) * patch_h
        grid_img.paste(patch, (x, y))

    return grid_img  # Return the final grid image


def show_filtered_keypoints(
    image, image_kpids, image_reference_coords, chosen_kpids, box_size=32
):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for kpid, (x, y) in zip(image_kpids, image_reference_coords):
        col = "green" if kpid in chosen_kpids else "red"
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=col)
        draw.text((x + 5, y - 5), str(kpid), fill="yellow")

        # Draw the rectangle (box) around the keypoint
        left, top, right, bottom = get_patch_boundary(image, [x, y], box_size)

        if kpid in chosen_kpids:
            draw.rectangle([left, top, right, bottom], outline=col)

    return image


def get_natural_sorted_filenames(directory):
    return natsorted(Path(directory).iterdir(), key=lambda f: f.name)

