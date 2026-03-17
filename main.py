import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import VBDemandDataset
from model import LNN

# ================= 训练配置参数 =================
EPOCHS = 60             # 训练轮数（本地测试先设少一点）
BATCH_SIZE = 4         # 每次喂给模型几个音频片段
LEARNING_RATE = 1e-3   # 学习率（决定模型每次修正错误的幅度）


NOISY_DIR = "archive/noisy_trainset_28spk_wav"
CLEAN_DIR = "archive/clean_trainset_28spk_wav"


def train():
    # 1. 确定计算设备 (如果有显卡就用 GPU，没有就用 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VBDemandDataset(NOISY_DIR, CLEAN_DIR, max_samples=None)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model=LNN().to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"\n--- 开始第 {epoch+1}/{EPOCHS} 轮训练 ---")
        for batch_idx, (noisy_batch, clean_batch) in enumerate(dataloader):
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            optimizer.zero_grad()
            mask=model(noisy_batch)
            predicted_clean=mask*noisy_batch
            loss=criterion(predicted_clean,clean_batch)
            loss.backward()
            optimizer.step()
            print(f"批次 [{batch_idx+1}/{len(dataloader)}] - 当前误差(Loss): {loss.item():.4f}")
    
    print("\n💾 训练完成！正在保存模型权重...")
    # 把模型内部所有的“齿轮参数”打包存进一个 .pth 文件里
    torch.save(model.state_dict(), "lnn_model.pth")
    print("✅ 模型已成功保存为 'lnn_model.pth'！")





if __name__ == "__main__":
    train()