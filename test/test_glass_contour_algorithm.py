import cv2
import os

# 从三个相机的合并图中抽取玻璃部分的轮廓

# root = "../data/241015_101025_566_4"
# root = "../data/241015_125935_725_16"
root = "../data/241016_110832_917_13"

image = cv2.imread(os.path.join(root, "kim", "original_image.png"), cv2.IMREAD_UNCHANGED)

# scaled_image = cv2.resize(image, None, fx=0.04, fy=0.04, interpolation=cv2.INTER_LINEAR)
# cv2.imshow("image", scaled_image)
# cv2.waitKey(0)

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

glass_image = image[y: image.shape[0] if y + h > image.shape[0] else y + h, x: image.shape[1] if x + w > image.shape[1] else x + w]
cv2.imwrite(os.path.join(root, "kim", "origin_glass_image.png"), glass_image)

scaled_image = cv2.resize(color_image, None, fx=0.04, fy=0.04, interpolation=cv2.INTER_LINEAR)
cv2.imshow("image", scaled_image)
cv2.waitKey(0)

print("finished")

