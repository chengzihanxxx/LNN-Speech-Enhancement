# 🌊 LNN Speech Enhancement (液体神经网络语音降噪)

本项目是一个基于 **PyTorch** 实现的语音降噪（Speech Enhancement）实验。核心网络采用前沿的 **Liquid Neural Networks (LNNs)**，探索其在处理非稳态音频噪声时的轻量化与时间序列建模能力。

## ✨ 项目亮点
* **轻量级高效：** 探索 LNN（如 CfC 模型）在极低参数量下对音频时序特征的捕捉。
* **端到端处理：** 包含完整的 STFT 频域特征提取、掩膜预测与 ISTFT 波形重建。
* **专业级可视化：** 包含 MATLAB 脚本，用于生成发表级的语谱图 (Spectrogram) 对比分析。

## 🚀 快速开始

### 1. 环境依赖
见requirements.txt

## 📂 数据集 (Datasets)

本项目模型在语音增强领域的标准数据集 **VoiceBank + DEMAND** 上进行训练与验证。

### 1. 数据集来源
* **下载地址：** (https://www.kaggle.com/datasets/jweiqi/voicebank-demand-16k)
* **内容简介：** 该数据集由干净的人声（VoiceBank）与多种环境噪音（DEMAND，如洗碗声、路边声、音乐、办公室底噪等）合成，包含不同信噪比（SNR）的训练集和测试集。

