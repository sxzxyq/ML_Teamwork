import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from torch.utils.data import DataLoader

# 引入之前的模块
from dataset import RetinopathyDataset, get_transforms
from models import DRModel

# ================= 配置 =================
class Config:
    CSV_PATH = "./data/processed/train_processed.csv"
    IMG_DIR = "./data/processed/train_images_512"
    # 指向刚才训练好的最佳模型权重
    MODEL_PATH = "./output/models/efficientnet_b3_fold1_best.pth"
    
    MODEL_NAME = 'efficientnet_b3'
    BATCH_SIZE = 16
    NUM_FOLDS = 5
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_val_data():
    """
    为了保证公平，必须使用和训练时一模一样的切分方式 (Seed=42)，
    只提取 Fold 1 的验证集。
    """
    df = pd.read_csv(Config.CSV_PATH)
    skf = StratifiedKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
    
    # 获取 Fold 1 (即第0次切分) 的验证集索引
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['diagnosis'])):
        if fold == 0:
            val_df = df.iloc[val_idx].reset_index(drop=True)
            return val_df
    return None

def main():
    print(f"正在加载 Fold 1 验证集...")
    val_df = load_val_data()
    print(f"验证集大小: {len(val_df)} 张图片")
    
    # 1. 准备 DataLoader
    val_dataset = RetinopathyDataset(
        csv_file=Config.CSV_PATH,
        root_dir=Config.IMG_DIR,
        transform=get_transforms(mode='val'),
        mode='train' # 依然是有标签模式
    )
    val_dataset.data = val_df # 覆盖数据
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. 加载模型
    print(f"正在加载模型: {Config.MODEL_PATH}")
    model = DRModel(model_name=Config.MODEL_NAME, pretrained=False) # 推理时不需要下载预训练权重
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.to(Config.DEVICE)
    model.eval()
    
    # 3. 开始预测
    preds = []
    targets = []
    
    print("正在进行推理 (Inference)...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(Config.DEVICE)
            outputs = model(images)
            
            # 收集结果
            preds.extend(outputs.view(-1).cpu().numpy())
            targets.extend(labels.numpy())
            
    # 4. 数据处理 (回归值 -> 类别)
    # 使用四舍五入
    preds_int = [int(round(max(0, min(4, x)))) for x in preds]
    targets_int = [int(x) for x in targets]
    
    # 5. 计算指标
    kappa = cohen_kappa_score(targets_int, preds_int, weights='quadratic')
    print(f"\n{'='*30}")
    print(f"最终验证集 Kappa 分数: {kappa:.4f}")
    print(f"{'='*30}\n")
    
    # 打印详细分类报告 (Precision, Recall, F1-Score)
    print("分类详细报告 (Classification Report):")
    print(classification_report(targets_int, preds_int, digits=4))
    
    # 6. 绘制混淆矩阵 (Confusion Matrix)
    # 这是报告里最重要的图！
    cm = confusion_matrix(targets_int, preds_int)
    # 归一化 (看百分比)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['0', '1', '2', '3', '4'],
                yticklabels=['0', '1', '2', '3', '4'])
    plt.xlabel('Predicted Label (预测值)')
    plt.ylabel('True Label (真实值)')
    plt.title(f'Confusion Matrix (Kappa: {kappa:.4f})')
    
    save_path = './output/confusion_matrix.png'
    plt.savefig(save_path)
    print(f"混淆矩阵已保存至: {save_path}")
    print("请打开该图片查看模型在各类别的详细表现。")
    plt.show()

if __name__ == "__main__":
    main()