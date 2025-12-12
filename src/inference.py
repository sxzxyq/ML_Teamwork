import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 引用模型
from models import DRModel

# ================= 配置 =================
class Config:
    # 你的测试集原始路径 (请确认你下载了 test_images)
    # 如果没有下载，可以用 validation集 模拟，这里先假设你有 test_images
    TEST_IMG_DIR = "./data/raw/test_images" 
    TEST_CSV_PATH = "./data/raw/test.csv"
    
    MODEL_PATH = "./output/models/efficientnet_b3_fold1_best.pth"
    OUTPUT_CSV = "./output/submission.csv"
    
    MODEL_NAME = 'efficientnet_b3'
    IMG_SIZE = 512
    BATCH_SIZE = 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= 1. 定义测试专用 Dataset =================
# 这里我们把 Member A 的 Ben Graham 预处理集成进来，做成"实时处理"
class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        # 只做归一化，不做旋转翻转
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.data)

    def crop_image_from_gray(self, img, tol=7):
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1),mask.any(0))]
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol
            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0): return img 
            else:
                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
                img = np.stack([img1,img2,img3],axis=-1)
        return img

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        # 尝试 png 或 jpg
        img_path = os.path.join(self.root_dir, f"{img_name}.png")
        if not os.path.exists(img_path):
             img_path = os.path.join(self.root_dir, f"{img_name}.jpg")

        image = cv2.imread(img_path)
        if image is None:
            # 容错处理
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # --- 实时 Ben Graham 处理 ---
            image = self.crop_image_from_gray(image)
            image = cv2.resize(image, (Config.IMG_SIZE, Config.IMG_SIZE))
            image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 10), -4, 128)
            # ---------------------------

        # 转 Tensor
        image = self.transform(image=image)['image']
        return image, img_name

def main():
    # 检查是否有测试数据
    if not os.path.exists(Config.TEST_IMG_DIR) or not os.path.exists(Config.TEST_CSV_PATH):
        print(f"⚠️ 警告: 找不到测试数据 ({Config.TEST_IMG_DIR})")
        print("如果是为了演示，这一步可以跳过。")
        print("或者你可以去下载 test_images.zip 放进去。")
        return

    print("开始生成测试集预测结果...")
    
    # 1. 准备数据
    dataset = TestDataset(Config.TEST_CSV_PATH, Config.TEST_IMG_DIR)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. 加载模型
    model = DRModel(model_name=Config.MODEL_NAME, pretrained=False)
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.to(Config.DEVICE)
    model.eval()
    
    predictions = []
    id_codes = []
    
    # 3. 推理
    with torch.no_grad():
        for images, ids in tqdm(loader):
            images = images.to(Config.DEVICE)
            outputs = model(images)
            
            # 回归值
            preds = outputs.view(-1).cpu().numpy()
            # 转整数类别 (0-4)
            preds_int = [int(round(max(0, min(4, x)))) for x in preds]
            
            predictions.extend(preds_int)
            id_codes.extend(ids)
            
    # 4. 保存结果
    submit_df = pd.DataFrame({
        'id_code': id_codes,
        'diagnosis': predictions
    })
    submit_df.to_csv(Config.OUTPUT_CSV, index=False)
    
    print(f"\n✅ 预测完成！")
    print(f"结果已保存至: {Config.OUTPUT_CSV}")
    print("你可以打开这个 CSV 查看每一张测试图的诊断结果。")

if __name__ == "__main__":
    main()