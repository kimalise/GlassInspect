import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


# 1. 定义模型
def load_model(model_path, num_classes=5):
    model = models.resnet18(pretrained=False)  # 初始化一个 ResNet-18
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # 修改输出层为类别数量

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载训练好的权重
    model.eval()  # 设置为评估模式
    return model


# 2. 定义图像预处理
def preprocess_image(image_path):
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.5,), std=(0.5,)),  # 适配灰度图
        ToTensorV2()
    ])

    # 读取灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # 应用预处理
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0)  # 增加 batch 维度
    return image_tensor


# 3. 预测函数
def predict_image(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        return predicted_class


# 4. 加载模型和类别名称
model_path = 'model/best_model.pth'  # 模型路径
class_names = ['Defect0', 'Defect1', 'Defect2', 'Defect3']  # 替换为你的类别名称
model = load_model(model_path, num_classes=len(class_names))

# 5. 加载图像并预测
image_path = 'data/split_dataset/val/2/0_03_414_212_.jpg'  # 输入图像路径
image_tensor = preprocess_image(image_path)
predicted_class = predict_image(model, image_tensor, class_names)

print(f"Predicted class: {predicted_class}")