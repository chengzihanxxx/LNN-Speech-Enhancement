import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class VBDemandDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, n_fft=1024, hop_length=256, seq_len=200, max_samples=10):
        """
        VoiceBank + DEMAND 数据集加载器 (工业级懒加载机制)
        
        参数:
            noisy_dir: 带噪语音文件夹路径
            clean_dir: 干净语音文件夹路径
            n_fft: STFT 窗口大小 (1024)
            hop_length: STFT 步长 (256)
            seq_len: 喂给 LNN 的固定时间序列长度 (200)
            max_samples: 本地调试时读取的最大文件数
        """
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.seq_len = seq_len
        self.window = torch.hann_window(n_fft) 
        
        # 1. 检查并获取文件列表
        if not os.path.exists(noisy_dir) or not os.path.exists(clean_dir):
            raise FileNotFoundError(f"找不到数据文件夹！\n当前尝试路径:\n{os.path.abspath(noisy_dir)}\n{os.path.abspath(clean_dir)}")
            
        noisy_files = set(f for f in os.listdir(noisy_dir) if f.endswith('.wav'))
        clean_files = set(f for f in os.listdir(clean_dir) if f.endswith('.wav'))
        valid_files = sorted(list(noisy_files.intersection(clean_files)))
        
        if len(valid_files) == 0:
            raise ValueError("没有找到匹配的 .wav 文件！请检查 archive 文件夹结构。")

        # 2. 限制数据量
        if max_samples is not None:
            self.files = valid_files[:max_samples]
            print(f"✅ [本地调试模式] 仅加载 {len(self.files)} 个音频对。")
        else:
            self.files = valid_files
            print(f"🔥 [全量训练模式] 准备加载全部 {len(self.files)} 个音频对。")

    def __len__(self):
        return len(self.files)

    def _process_audio(self, file_path):
        """将音频转换为 [Time, Features] 的幅度谱矩阵"""
        waveform, sample_rate = torchaudio.load(file_path)
        
        # 统一采样率
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # STFT 变换
        stft_matrix = torch.stft(
            waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window, 
            return_complex=True
        )
        
        # 取幅度谱并转置为 [Time, Features]
        mag = torch.abs(stft_matrix).squeeze(0).transpose(0, 1)
        
        # 对齐长度到 seq_len
        if mag.shape[0] < self.seq_len:
            mag = torch.nn.functional.pad(mag, (0, 0, 0, self.seq_len - mag.shape[0]))
        else:
            mag = mag[:self.seq_len, :]
            
        return mag

    def __getitem__(self, idx):
        filename = self.files[idx]
        noisy_mag = self._process_audio(os.path.join(self.noisy_dir, filename))
        clean_mag = self._process_audio(os.path.join(self.clean_dir, filename))
        return noisy_mag, clean_mag

if __name__ == "__main__":
    # 因为文件现在就在根目录，所以路径直接写 archive/... 即可
    noisy_dir = "archive/noisy_trainset_28spk_wav"
    clean_dir = "archive/clean_trainset_28spk_wav"
    
    dataset = VBDemandDataset(noisy_dir, clean_dir, max_samples=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("\n📦 DataLoader 准备就绪，正在测试抓取数据...")
    
    try:
        noisy_batch, clean_batch = next(iter(dataloader))
        print(f"\n✅ 数据加载成功！")
        print(f"🎙️ 输入形状 (Batch, Time, Freq): {noisy_batch.shape}")
        print(f"🎯 目标形状 (Batch, Time, Freq): {clean_batch.shape}")
    except Exception as e:
        print(f"\n❌ 出错了: {e}")