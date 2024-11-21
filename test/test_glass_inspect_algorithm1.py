import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 以下算法模仿了halcon的dyn_threshold函数
# 基本思路是判断原始图像和均值滤波后的图像之间的像素差异是否大于某一个阈值,如果是则认为是缺陷区域

root = "../data/241015_125935_725_16"
# root = "../data/241015_101025_566_4"

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

min_area = 100
max_area = 100000

selected_regions = []
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if min_area <= area:
        selected_regions.append(i)

selected_image = np.zeros_like(dark_pixels, dtype=np.uint8)
for region in selected_regions:
    selected_image[labels == region] = 255

# 绘制矩形框
# contours, _ = cv2.findContours(selected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 创建彩色图像以便绘制结果
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# 遍历每个轮廓，计算最小外接矩形
for region in selected_regions:
    selected_image_1 = np.zeros_like(dark_pixels, dtype=np.uint8)
    selected_image_1[labels == region] = 255
    cs, _ = cv2.findContours(selected_image_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算最小外接矩形
    rect = cv2.minAreaRect(cs[0])

    # 获取矩形的4个顶点坐标
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # 转换为整数

    # 绘制矩形
    cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)

print("finished")

