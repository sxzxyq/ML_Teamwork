import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

# ================= 配置 =================
# 你的绝对路径
RAW_DATA_DIR = "C:/Users/23166/Desktop/code/ML_Teamwork/data/raw/train_images"
RAW_CSV_PATH = "C:/Users/23166/Desktop/code/ML_Teamwork/data/raw/train.csv"
SAVE_DIR = "./data/processed/train_images_512" 
SAVE_CSV_PATH = "./data/processed/train_processed.csv" 

IMG_SIZE = 512
# =======================================

def get_class_weights(df):
    """计算类别权重"""
    if 'diagnosis' not in df.columns:
        print("CSV中找不到 diagnosis 列，跳过权重计算。")
        return
    counts = df['diagnosis'].value_counts().sort_index()
    print("\n=== 样本分布统计 ===")
    print(counts)

def crop_image_from_gray(img, tol=7):
    """切黑边"""
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

def ben_graham_process(img_path):
    """Ben Graham's Method"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
        return img
    except Exception as e:
        return None

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # 1. 读取原始 CSV
    raw_df = pd.read_csv(RAW_CSV_PATH)
    # 将 id_code 设为索引，方便快速查找
    raw_df.set_index('id_code', inplace=True)
    
    # 2. 获取文件夹里实际所有的图片
    # 支持 png, jpg, jpeg
    all_files = glob(os.path.join(RAW_DATA_DIR, "*"))
    all_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n文件夹中实际找到 {len(all_files)} 张图片，开始处理...")
    
    valid_data = []
    
    for img_path in tqdm(all_files):
        # 获取文件名 (例如: 001639a390f0_aug0.png)
        file_name = os.path.basename(img_path)
        # 获取不带后缀的ID (例如: 001639a390f0_aug0)
        file_id = os.path.splitext(file_name)[0]
        
        # ★关键逻辑：尝试还原真实ID★
        # 如果文件名里包含 "_aug"，我们把它去掉来去 CSV 里找标签
        real_id = file_id.split('_aug')[0] 
        
        # 在 CSV 中查找标签
        try:
            diagnosis = raw_df.loc[real_id, 'diagnosis']
            # 如果 raw_df 中有重复ID，loc可能会返回 Series，取第一个
            if isinstance(diagnosis, pd.Series):
                diagnosis = diagnosis.iloc[0]
        except KeyError:
            # print(f"警告：图片 {file_name} 在 CSV 中找不到标签，跳过。")
            continue

        # 如果已经处理过，直接记录
        dst_path = os.path.join(SAVE_DIR, f"{file_id}.png")
        if os.path.exists(dst_path):
            valid_data.append([file_id, diagnosis])
            continue
            
        # 处理图片
        processed_img = ben_graham_process(img_path)
        
        if processed_img is not None:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_path, processed_img)
            # 记录文件名(ID)和标签
            valid_data.append([file_id, diagnosis])
            
    # 保存新的 CSV
    new_df = pd.DataFrame(valid_data, columns=['id_code', 'diagnosis'])
    new_df.to_csv(SAVE_CSV_PATH, index=False)
    
    # 简单的 EDA
    get_class_weights(new_df)
    
    print(f"\n预处理完成！成功处理并匹配标签的图片数: {len(new_df)}")
    print(f"处理后图片保存在: {SAVE_DIR}")

if __name__ == "__main__":
    main()