import torch
import torch.nn as nn
from ncps.torch import CfC  # 导入我们在要求里安装的 LNN 核心层

class LNN(nn.Module):
    def __init__(self, input_size=513, hidden_size=64, output_size=513):
       
        super(LNN, self).__init__()
        
        # 1. 核心 LNN 层 (CfC)：负责处理时间序列，记住过去的噪声规律
        # batch_first=True 意思是我们的数据形状第一维是 Batch [Batch, Time, Features]
        self.model = CfC(input_size, hidden_size, batch_first=True)
        
        # 2. 全连接层 (Linear)：把 CfC 思考后的隐藏状态，映射回我们需要的输出维度 (513)
        self.linear = nn.Linear(hidden_size, output_size)
        
        # 3. 激活函数 (Sigmoid)：把输出的数值压缩到 0 ~ 1 之间，变成一个“百分比滤网(Mask)”
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 💡 PyTorch 推荐直接调用实例 self.model(x)，而不是 self.model.forward(x)
        out, hidden = self.model(x)
        mask = self.sigmoid(self.linear(out))
        return mask


# 测试代码骨架
if __name__ == "__main__":
    # 创建一个模拟的输入张量 (Batch=4, Time=200, Freq=513)
    # 这正好对应我们 data_loader.py 输出的形状
    dummy_input = torch.randn(4, 200, 513)
    
    # 实例化模型
    model = LNN()
    
    # 前向传播测试
    output_mask = model(dummy_input)
    
    print(f"🎙️ 输入特征形状: {dummy_input.shape}")
    print(f"🎯 输出掩码形状: {output_mask.shape}")
    print("🎉 如果形状一致，说明模型的 '任督二脉' 已经彻底打通！")