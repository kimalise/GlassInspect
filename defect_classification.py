import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import cv2
from torchvision import models
import torch.nn as nn
from torch.optim import Adam

data_dir = "data/split_dataset"
num_classes = 4
batch_size = 32
num_epochs = 100
learning_rate = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Albumentations 数据增强定义
train_transforms = A.Compose([
    A.Resize(224, 224),
    A.RandomBrightnessContrast(p=0.2),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.Normalize(mean=(0.5), std=(0.5)),
    ToTensorV2(),  # 转换为 PyTorch Tensor
])

val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5), std=(0.5)),
    ToTensorV2(),
])


# 3. 自定义 Dataset
class CustomDataset(Dataset):
    def __init__(self, data_dir, phase, transform=None):
        self.data_dir = os.path.join(data_dir, phase)
        self.transform = transform
        self.classes = os.listdir(self.data_dir)
        self.samples = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, img_name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

# 加载数据集
train_dataset = CustomDataset(data_dir, 'train', transform=train_transforms)
val_dataset = CustomDataset(data_dir, 'val', transform=val_transforms)

# 计算每个样本的权重
class_counts = [31, 60, 29, 8]  # 假设每个类别的样本数量
class_weights = [1.0 / count for count in class_counts]
sample_weights = [class_weights[label] for _, label in train_dataset.samples]

# 创建采样器
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# 4. 定义模型
model = models.resnet18(pretrained=True)  # 使用 ResNet-18 作为骨架网络
# 修改第一层卷积层以支持单通道输入
model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# # 冻结所有参数
# for param in model.parameters():
#     param.requires_grad = False
#
# # 解冻 conv1 和 fc 层
# for param in model.conv1.parameters():
#     param.requires_grad = True
# for param in model.fc.parameters():
#     param.requires_grad = True

model = model.to(device)

# 5. 定义损失函数和优化器
class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float32)
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# 6. 训练和验证函数
def train_model(model, criterion, optimizer, num_epochs=25):
    best_acc = -1
    best_epoch = -1
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch

                    # 8. 保存模型
                    torch.save(model.state_dict(), 'model/best_model.pth')
    print("Training complete, best val acc: {:.4f} epoch: {}".format(best_acc, best_epoch))
    return model

# 7. 开始训练
trained_model = train_model(model, criterion, optimizer, num_epochs=num_epochs)

