import cv2
import os
import numpy as np
import json

# image_root = "../data/241015_094222_820_2/images"
# image_root = "../data/241015_125935_725_16/images"
# root = "../data/241015_101025_566_4"
root = "../data/241015_125935_725_16"

def read_sample_info(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

images = os.listdir(os.path.join(root, "images"))

camera_images = {}
cur_cam_idx = -1
for image in images:
    # one_camera_images = []
    cam_idx, filename = image.split("_")

    if cam_idx not in camera_images:
        camera_images[cam_idx] = []

    img = cv2.imread(os.path.join(root, "images", image), cv2.IMREAD_UNCHANGED)
    camera_images[cam_idx].append(img)

# 拼接
image_for_each_camera = []
for cam_idx, cam_images in camera_images.items():
    image_list = camera_images[cam_idx]
    cam_image_stacked = np.vstack(image_list)
    image_for_each_camera.append(cam_image_stacked)

min_row = np.min([img.shape[0] for img in image_for_each_camera], axis=0)
image_for_each_camera = [img[:min_row, :] for img in image_for_each_camera]

# full_image = np.hstack(image_for_each_camera) # 直接拼接的话多个相机采集的图像之间是有重叠的地方

sample_info = read_sample_info(os.path.join(root, "sample_info.json"))
pixel_range = sample_info["device"]["pixel_range"]

image_for_each_camera = [img[:, pixel_range[idx]['first']:pixel_range[idx]['second']] for idx, img in enumerate(image_for_each_camera)]
full_image = np.hstack(image_for_each_camera) # 直接拼接的话多个相机采集的图像之间是有重叠的地方

full_image = cv2.rotate(full_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imwrite(os.path.join(root, "kim", "original_image.png"), full_image)

print("finish")