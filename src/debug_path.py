import os

# 填入你代码里写的路径
TARGET_DIR = "./data/raw/train_images"  # 或者改成你的真实路径

print(f"当前工作目录 (CWD): {os.getcwd()}")
abs_path = os.path.abspath(TARGET_DIR)
print(f"代码试图寻找的绝对路径: {abs_path}")

if os.path.exists(abs_path):
    print("✅ 文件夹存在！")
    files = os.listdir(abs_path)
    print(f"文件夹里有 {len(files)} 个文件。")
    if len(files) > 0:
        print(f"前3个文件: {files[:3]}")
    else:
        print("❌ 但是文件夹是空的！")
else:
    print("❌ 文件夹不存在！请检查路径拼写。")