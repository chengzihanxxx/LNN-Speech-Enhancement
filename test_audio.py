import torch
import torchaudio
import scipy.io as sio  # 用来导出给 MATLAB 的 .mat 文件
import os

from model import LNN

# ================= 测试配置参数 =================
MODEL_PATH = "lnn_model.pth"
# 把路径改成测试集 (testset) 里的某一个音频
TEST_NOISY_WAV = "archive/noisy_testset_wav/p232_001.wav" 
TEST_CLEAN_WAV = "archive/clean_testset_wav/p232_001.wav"
OUTPUT_WAV = "denoised_output.wav"
OUTPUT_MAT = "matlab_data.mat"

# STFT 参数（必须和训练时保持绝对一致！）
N_FFT = 1024
HOP_LENGTH = 256
# ===============================================

def test_and_reconstruct():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 测试设备: {device}")

    # ========================================================
   
    # ========================================================
    model = LNN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ 模型记忆加载成功！")



    # ========================================================
    
    # ========================================================
    print("\n🎧 正在读取音频文件...")
    noisy_waveform, sr = torchaudio.load(TEST_NOISY_WAV)
    clean_waveform, _ = torchaudio.load(TEST_CLEAN_WAV)

    window = torch.hann_window(N_FFT)
    noisy_stft = torch.stft(noisy_waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
    clean_stft = torch.stft(clean_waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)

    noisy_mag = torch.abs(noisy_stft)
    clean_mag = torch.abs(clean_stft)
    noisy_phase = torch.angle(noisy_stft)
    print("✅ 音频特征解剖完毕！")




    # ========================================================
    
    # ========================================================
    print("\n🧠 AI 正在进行降噪")
    # 1. 翻转矩阵适应模型: [1, 513, Time] -> [1, Time, 513] 并送入设备
    mag_input = noisy_mag.permute(0, 2, 1).to(device)

    # 2. 预测并生成滤网 (用 torch.no_grad() 告诉 PyTorch 不需要计算梯度，省内存提速！)
    with torch.no_grad():
        mask = model(mag_input)
        print(f"🔍 检查 Mask 极值 -> 最大值: {mask.max().item():.4f}, 最小值: {mask.min().item():.4f}")
        print(f"🔍 检查 Mask 平均值 -> {mask.mean().item():.4f}")
        pred_mag = mag_input * mask  # 滤除噪音

    
    pred_mag = pred_mag.permute(0, 2, 1).cpu()
    print("✅ 手术成功！已剥离噪声。")




    # ========================================================
   
    # ========================================================
    print("\n🪄 正在将数据重组成声音...")
    # 1. 使用极坐标函数 (polar)，把干净的幅度和原始的相位“拼”回完整的复数矩阵
    pred_stft = torch.polar(pred_mag, noisy_phase)

    # 2. 逆变换 (istft)，把二维矩阵变回一维的声音波形
    pred_waveform = torch.istft(pred_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window)

    # 3. 保存成能听的 .wav 文件
    torchaudio.save(OUTPUT_WAV, pred_waveform, sr)
    print(f"🎵 太棒了！降噪后的音频已保存为: {OUTPUT_WAV}")




    # ========================================================
    
    # ========================================================
    # 把张量(Tensor)去掉多余的维度(squeeze)并转成 numpy 数组，打包存入 .mat
    mat_dict = {
        'noisy_mag': noisy_mag.squeeze().numpy(),
        'clean_mag': clean_mag.squeeze().numpy(),
        'pred_mag': pred_mag.squeeze().numpy(),
        'noisy_wav': noisy_waveform.squeeze().numpy(),
        'clean_wav': clean_waveform.squeeze().numpy(),
        'pred_wav': pred_waveform.squeeze().numpy()
    }
    sio.savemat(OUTPUT_MAT, mat_dict)
    print(f"📊 实验数据已打包！你可以用 MATLAB 打开 '{OUTPUT_MAT}' 绘制炫酷的论文对比图了！")

if __name__ == "__main__":
    test_and_reconstruct()