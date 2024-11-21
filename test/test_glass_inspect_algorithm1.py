import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# root = "../data/241015_125935_725_16"
root = "../data/241015_101025_566_4"

# 1. 读取图像
# image_path = '/mnt/data/origin_glass_image.png'
image_path = os.path.join(root, "kim", "origin_glass_image.png")
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

kernel_size = 30
image_mean = cv2.blur(image, (kernel_size, kernel_size))

offset = 8
dynamic_threshold = image_mean - offset

dark_pixels = (image < dynamic_threshold).astype(np.uint8) * 255

# 形态学: 膨胀,目的是为了把临近的多个区域合并
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dark_pixels = cv2.dilate(dark_pixels, dilate_kernel, iterations=1)

# num_labels, labels = cv2.connectedComponents(dark_pixels)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_pixels)

min_area = 400
max_area = 100000

selected_regions = []
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if min_area <= area:
        selected_regions.append(i)

selected_image = np.zeros_like(dark_pixels, dtype=np.uint8)
for region in selected_regions:
    selected_image[labels == region] = 255

print("finished")

