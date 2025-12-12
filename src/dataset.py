import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train'):
        """
        Args:
            csv_file (string): CSV 文件路径
            root_dir (string): 图片文件夹路径
            transform (callable, optional): 图像增强转换
            mode (string): 'train' 或 'val'。如果是 'test'，则没有标签。
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 获取图片文件名
        # 假设 csv 里第一列是 id_code
        img_name = self.data.iloc[idx, 0]
        # 加上后缀 (我们预处理保存的是 .png)
        img_path = os.path.join(self.root_dir, f"{img_name}.png")

        # 2. 读取图片
        image = cv2.imread(img_path)
        if image is None:
            # 如果读不到图（极少数情况），生成一张全黑图防止报错
            # 这里的 512 要和你预处理的尺寸一致
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            print(f"Warning: Could not read image {img_path}")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. 应用数据增强 (Augmentation)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # 如果没有定义 transform，至少要转成 Tensor
            base_transform = A.Compose([A.Normalize(), ToTensorV2()])
            image = base_transform(image=image)['image']

        # 4. 返回结果
        # 如果是 Test 模式，只返回图片
        if self.mode == 'test':
            return image
        
        # 如果是 Train/Val 模式，返回 (图片, 标签)
        # 注意：我们要用回归(Regression)的方式训练，所以标签转为 float
        label = self.data.iloc[idx, 1]
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, label

# ==========================================
# 定义数据增强策略 (Member A 的策略在这里落地)
# ==========================================
def get_transforms(img_size=512, mode="train"):
    if mode == "train":
        return A.Compose([
            # 随机旋转 (解决样本不足的核心手段)
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            # 随机改变亮度/对比度 (模拟不同医院的设备差异)
            A.RandomBrightnessContrast(p=0.5),
            # 归一化 (使用 ImageNet 标准均值方差)
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        # 验证集/测试集：不做增强，只做归一化
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

# ==========================================
# 简单的测试代码 (组长验证用)
# ==========================================
if __name__ == "__main__":
    # 配置路径 (请修改为你自己的)
    CSV_PATH = "./data/processed/train_processed.csv"
    IMG_DIR = "./data/processed/train_images_512"
    
    # 1. 实例化 Dataset
    transforms = get_transforms(mode="train")
    dataset = RetinopathyDataset(csv_file=CSV_PATH, root_dir=IMG_DIR, transform=transforms)
    
    print(f"数据集长度: {len(dataset)}")
    
    # 2. 测试读取一张
    img, label = dataset[0]
    print(f"图片 Tensor 形状: {img.shape}") # 应该是 [3, 512, 512]
    print(f"标签: {label} (类型: {label.dtype})")
    print("Dataset 测试通过！可以交给 DataLoader 了。")