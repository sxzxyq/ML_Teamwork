import torch
import timm
import cv2
import sklearn
import pandas as pd

print("="*30)
print("环境自检报告")
print("="*30)
print(f"1. PyTorch 版本: {torch.__version__}")
print(f"2. CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"   显卡数量: {torch.cuda.device_count()}")
else:
    print("   警告: 未检测到 GPU，训练将非常缓慢！")

print(f"3. Pandas 版本: {pd.__version__}")
print(f"4. Scikit-learn 版本: {sklearn.__version__}")
print(f"5. OpenCV 版本: {cv2.__version__}")
print(f"6. Timm (模型库) 版本: {timm.__version__}")
print("="*30)
print("环境配置成功，可以开始干活了！")