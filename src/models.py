import torch
import torch.nn as nn
import timm

class DRModel(nn.Module):
    def __init__(self, model_name='efficientnet_b3', pretrained=True):
        """
        Args:
            model_name (str): timm 支持的模型名称 (如 'efficientnet_b3', 'resnet50')
            pretrained (bool): 是否加载 ImageNet 预训练权重 (迁移学习核心)
        """
        super(DRModel, self).__init__()
        
        # 1. 加载骨干网络 (Backbone)
        # num_classes=0 表示去掉原始的分类头 (我们自己加)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # 2. 获取骨干网络的输出特征维度
        # efficientnet_b3 的特征维度通常是 1536
        in_features = self.backbone.num_features
        
        # 3. 定义新的头部 (Head)
        # 我们用回归方式 (Regression)，所以输出节点数是 1
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),            # 激活函数
            nn.Dropout(p=0.3),    # 防止过拟合
            nn.Linear(512, 1)     # 最终输出一个浮点数 (0.0 ~ 4.0)
        )

    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        # 预测结果
        output = self.head(features)
        return output

# ==========================================
# 组长验证区域
# ==========================================
if __name__ == "__main__":
    # 模拟一个 Batch 的数据 (Batch_Size=2, Channel=3, Height=512, Width=512)
    dummy_input = torch.randn(2, 3, 512, 512)
    
    print("正在下载/加载预训练模型，请稍候...")
    model = DRModel(model_name='efficientnet_b3', pretrained=True)
    
    # 试运行一次前向传播
    output = model(dummy_input)
    
    print("\n=== 模型测试报告 ===")
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape} (预期应该是 [2, 1])")
    print(f"模型输出示例: {output.detach().numpy().flatten()}")
    print("✅ 模型构建成功！Member C 任务完成。")