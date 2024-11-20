import cv2
import os
import numpy as np

# image_root = "../data/241015_094222_820_2/images"
image_root = "../data/241015_101025_566_4/images"

images = os.listdir(image_root)

camera_images = {}
cur_cam_idx = -1
for image in images:
    # one_camera_images = []
    cam_idx, filename = image.split("_")
    if cam_idx in camera_images:
        img = cv2.imread(os.path.join(image_root, image), cv2.IMREAD_UNCHANGED)
        camera_images[cam_idx].append(img)
    else:
        camera_images[cam_idx] = []

# 拼接
image_for_each_camera = []
for cam_idx, cam_images in camera_images.items():
    image_list = camera_images[cam_idx]
    cam_image_stacked = np.vstack(image_list)
    image_for_each_camera.append(cam_image_stacked)

min_row = np.min([img.shape[0] for img in image_for_each_camera], axis=0)
image_for_each_camera = [img[:min_row, :] for img in image_for_each_camera]

full_image = np.hstack(image_for_each_camera) # 直接拼接的话多个相机采集的图像之间是有重叠的地方

print("finish")