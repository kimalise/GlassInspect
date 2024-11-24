# 检测算法的通用模块
import cv2
import numpy as np
import os

def merge_images(camera_images, image_range):
    '''
    :param camera_images: {cam_id: [image_frame0, image_frame1, ...]}, 其中image_frame0, image_frame1是该相机中采集的图像帧
    :param image_range: 图像合并时的边界 {cam_id: {'first': xxx, 'second': yyy}}
    :return: 合并之后的一个图像
    '''

    # 拼接
    image_for_each_camera = []
    for cam_idx, cam_images in camera_images.items():
        image_list = camera_images[cam_idx]
        cam_image_stacked = np.vstack(image_list)
        image_for_each_camera.append(cam_image_stacked)

    min_row = np.min([img.shape[0] for img in image_for_each_camera], axis=0) # 拼接之后的每个相机的图像宽度是相同，比如都是1024，但是高度可能有1-2个像素的差别
    image_for_each_camera = [img[:min_row, :] for img in image_for_each_camera]

    image_for_each_camera = [img[:, image_range[idx]['first']:image_range[idx]['second']] for idx, img in
                             enumerate(image_for_each_camera)]
    full_image = np.hstack(image_for_each_camera)  # 直接拼接的话多个相机采集的图像之间是有重叠的地方

    full_image = cv2.rotate(full_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return full_image

def extract_glass_area(image):
    '''
    :param image: 多相机图像合并之后的大图
    :return: 玻璃区域图像，在玻璃边界的基础上往外扩充一定的距离
    '''
    # image = cv2.imread(os.path.join(root, "kim", "original_image.png"), cv2.IMREAD_UNCHANGED)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    _, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 10000000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 100)

    # 保存玻璃区域图像
    border = 1450
    x = x - border
    y = y - border
    w = w + border * 2
    h = h + border * 2
    x = 0 if x < 0 else x
    y = 0 if y < 0 else y

    glass_image = image[y: image.shape[0] if y + h > image.shape[0] else y + h,
                  x: image.shape[1] if x + w > image.shape[1] else x + w]

    return glass_image

def merge_rotated_rectangles(rectangles):
    def are_rects_intersecting(rect1, rect2):
        """检测两个旋转矩形是否相交"""
        intersection, points = cv2.rotatedRectangleIntersection(rect1, rect2)
        return intersection != cv2.INTERSECT_NONE

    # 将矩形框分组
    groups = []
    for rect in rectangles:
        merged = False
        for group in groups:
            if any(are_rects_intersecting(rect, other) for other in group):
                group.append(rect)
                merged = True
                break
        if not merged:
            groups.append([rect])

    # 计算每组的最小外接旋转矩形
    merged_rectangles = []
    for group in groups:
        points = []
        for rect in group:
            box_points = cv2.boxPoints(rect)  # 获取旋转矩形的顶点
            points.extend(box_points)
        points = np.array(points)
        merged_rect = cv2.minAreaRect(points)  # 计算最小外接旋转矩形
        merged_rectangles.append(merged_rect)

    return merged_rectangles

def inspect_glass_defect1(glass_image):
    # image_path = os.path.join(root, "kim", "origin_glass_image.png")
    # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    kernel_size = 30
    image_mean = cv2.blur(glass_image, (kernel_size, kernel_size))

    offset = 8
    dynamic_threshold = image_mean - offset

    dark_pixels = (glass_image < dynamic_threshold).astype(np.uint8) * 255

    # 形态学: 膨胀,目的是为了把临近的多个区域合并
    # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    dark_pixels = cv2.dilate(dark_pixels, dilate_kernel, iterations=1)

    # num_labels, labels = cv2.connectedComponents(dark_pixels)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_pixels)

    min_area = 300
    # min_area = 50
    max_area = 1000000

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
    output_image = cv2.cvtColor(glass_image, cv2.COLOR_GRAY2BGR)

    # 遍历每个轮廓，计算最小外接矩形
    rects = []
    for region in selected_regions:
        selected_image_1 = np.zeros_like(dark_pixels, dtype=np.uint8)
        selected_image_1[labels == region] = 255
        cs, _ = cv2.findContours(selected_image_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 这里再次过滤一下，把面积较大的轮廓点去掉
        area = cv2.contourArea(cs[0])
        if area >= max_area:
            continue

        # 计算最小外接矩形
        rect = cv2.minAreaRect(cs[0])
        rects.append(rect)

    rects = merge_rotated_rectangles(rects)

    for rect in rects:
        # 获取矩形的4个顶点坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)  # 转换为整数

        # 绘制矩形
        cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)

    return output_image
