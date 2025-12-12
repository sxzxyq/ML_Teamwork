import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

# 引入我们之前写好的模块
from dataset import RetinopathyDataset, get_transforms
from models import DRModel

# ==========================================
# 0. 超参数配置 (Config)
# ==========================================
class Config:
    CSV_PATH = "./data/processed/train_processed.csv"
    IMG_DIR = "./data/processed/train_images_512"
    SAVE_DIR = "./output/models"
    
    MODEL_NAME = 'efficientnet_b3'
    IMG_SIZE = 512
    BATCH_SIZE = 8       # 显存如果不够(比如小于6G)，改成 4
    EPOCHS = 10           # 先跑 10 轮试试水 (实际比赛建议 30+)
    LR = 1e-4            # 学习率
    NUM_FOLDS = 5        # 5折交叉验证
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 确保输出目录存在
os.makedirs(Config.SAVE_DIR, exist_ok=True)

# ==========================================
# 1. 辅助函数
# ==========================================
def seed_everything(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    # 进度条
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # 回归 Loss: MSE (均方误差)
        # 注意：outputs 是 [batch, 1], labels 也是 [batch]，需要对齐维度
        loss = criterion(outputs.view(-1), labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs.view(-1), labels)
            running_loss += loss.item()
            
            # 收集预测结果用于计算 Kappa
            preds.extend(outputs.view(-1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    # === 关键步骤：回归值 -> 分类整数 ===
    # 简单的四舍五入策略: 
    # <0.5 -> 0, 0.5~1.5 -> 1, ... >3.5 -> 4
    preds_int = [int(round(max(0, min(4, x)))) for x in preds]
    targets_int = [int(x) for x in targets]
    
    # 计算 Quadratic Weighted Kappa
    kappa = cohen_kappa_score(targets_int, preds_int, weights='quadratic')
    
    return running_loss / len(loader), kappa

# ==========================================
# 2. 主训练循环
# ==========================================
def main():
    seed_everything(Config.SEED)
    
    # 读取全部数据
    df = pd.read_csv(Config.CSV_PATH)
    
    # Stratified K-Fold (分层抽样，解决样本不平衡的关键)
    skf = StratifiedKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
    
    # 我们只演示跑 第1折 (Fold 0)，节省时间
    # 如果想跑全量，可以在外面加一个 for fold in range(Config.NUM_FOLDS)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['diagnosis'])):
        print(f"\n{'='*20} Fold {fold+1}/{Config.NUM_FOLDS} {'='*20}")
        
        # 1. 划分数据
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # 2. 建立 Dataset & DataLoader
        train_dataset = RetinopathyDataset(
            csv_file=Config.CSV_PATH, # 这里为了偷懒，dataset类里还是读csv。实际应该把df传进去，这需要微调dataset类。
            # 为了简单，我们临时保存分折后的csv (生产环境可以优化)
            root_dir=Config.IMG_DIR,
            transform=get_transforms(mode='train'),
            mode='train'
        )
        # 修正: 直接重写 Dataset.__init__ 比较麻烦，
        # 我们可以简单地把 train_df 和 val_df 覆盖进去
        train_dataset.data = train_df 
        
        val_dataset = RetinopathyDataset(
            csv_file=Config.CSV_PATH, 
            root_dir=Config.IMG_DIR,
            transform=get_transforms(mode='val'), # 验证集不做增强
            mode='train'
        )
        val_dataset.data = val_df
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0) # win下建议 workers=0
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
        
        # 3. 初始化模型
        model = DRModel(model_name=Config.MODEL_NAME, pretrained=True)
        model = model.to(Config.DEVICE)
        
        # 4. 定义优化器和损失
        # 回归任务用 MSELoss
        criterion = nn.MSELoss() 
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-5)
        
        # 学习率调度器 (Cosine Annealing)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
        
        best_kappa = -1.0
        
        # 5. Epoch 循环
        for epoch in range(Config.EPOCHS):
            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
            
            # Val
            val_loss, val_kappa = validate(model, val_loader, criterion, Config.DEVICE)
            
            # Step Scheduler
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Kappa: {val_kappa:.4f}")
            
            # 保存最佳模型
            if val_kappa > best_kappa:
                best_kappa = val_kappa
                save_path = f"{Config.SAVE_DIR}/{Config.MODEL_NAME}_fold{fold+1}_best.pth"
                torch.save(model.state_dict(), save_path)
                print(f"    >>> Kappa Improved! Model Saved: {save_path}")
        
        print(f"\nFold {fold+1} Finished. Best Kappa: {best_kappa:.4f}")
        
        # 演示目的，只跑一折就退出。
        # 如果你想跑完所有折，把下面这行去掉
        break 

if __name__ == "__main__":
    main()