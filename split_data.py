import os
import shutil
from sklearn.model_selection import train_test_split


def split_dataset(root_dir, output_dir, val_ratio=0.2, random_seed=42):
    """
    将数据集按照一定比例划分为训练集和验证集。

    Args:
        root_dir (str): 数据集的根目录，包含类别文件夹。
        output_dir (str): 输出数据集的目录，划分后的数据集存储在此目录。
        val_ratio (float): 验证集所占比例（默认 0.2）。
        random_seed (int): 随机种子（保证可重复）。
    """
    # 设置输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # 遍历根目录的每个类别文件夹
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # 获取该类别下所有文件
        file_list = os.listdir(class_path)
        file_list = [os.path.join(class_path, f) for f in file_list if os.path.isfile(os.path.join(class_path, f))]

        # 按比例划分训练集和验证集
        train_files, val_files = train_test_split(file_list, test_size=val_ratio, random_state=random_seed)

        # 创建类别目录
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # 移动文件
        for file in train_files:
            shutil.copy(file, os.path.join(train_class_dir, os.path.basename(file)))
        for file in val_files:
            shutil.copy(file, os.path.join(val_class_dir, os.path.basename(file)))

        print(f"Class {class_name}: {len(train_files)} training samples, {len(val_files)} validation samples.")


# 使用示例
root_dir = 'data/dataset_defect_classification'  # 原始数据集路径，包含类别文件夹
output_dir = 'data/split_dataset'  # 划分后数据集保存路径
split_dataset(root_dir, output_dir, val_ratio=0.2)