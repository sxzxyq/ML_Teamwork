
## 环境配置（一键复现）

请先安装 [Anaconda](https://www.anaconda.com/) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

```bash
# 1. 创建环境
conda create -n team_project python=3.9 -y

# 2. 激活环境
conda activate team_project

# 3. 安装 PyTorch + CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. 安装其他依赖
pip install opencv-python pandas matplotlib seaborn scikit-learn timm

# 5. 安装指定版本 albumentations
pip install albumentations==1.3.1
