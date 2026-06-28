# VoxCPM 2 LJSpeech 微调详细指南

虚拟环境
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## 步骤 1 — 生成 JSONL 清单文件 (Manifest)

```bash
python scripts/convert_ljspeech.py -i <lj_speech_metadata_path> 
```

生成的清单文件内容示例如下：

```json
{"audio": "/path/to/LJSpeech-1.1/wavs/LJ001-0001.wav", "text": "Chapter one missus rachel lynde is surprised ..."}
{"audio": "/path/to/LJSpeech-1.1/wavs/LJ001-0002.wav", "text": "That had its source away back in the woods of the old cuthbert place ..."}
```

## 步骤 2 — 数据预处理

为了极大提升训练速度并节省显存，强烈建议在训练前将音频离线提取为 Latent 特征。
运行配套的预处理脚本：

```bash
python scripts/preprocess_latents.py \
    -i train.jsonl \
    -o train_latents.jsonl \
    -d ./extracted_latents \
    -p openbmb/VoxCPM2
```

该脚本会自动完成：
1. 将任意采样率的音频自动重采样至 16000Hz。
2. 使用 AudioVAE 提取特征，并将其保存为 `.pt` 张量文件到 `./extracted_latents` 目录下。
3. 生成全新的 `train_latents.jsonl` 清单文件，内部自动附带 `is_latent: true` 标记。

训练脚本一旦检测到该标记，就会自动切换到 **Latent 模式**，彻底跳过音频解码和在线 Encode 阶段，极大降低显存占用并提升训练速度！

> 后续的微调配置文件中，请务必将 `train_manifest` 替换为新生成的 `train_latents.jsonl`。

## 步骤 3a — 全参数微调 (SFT)

```bash
# 单卡 GPU 训练
python scripts/train_voxcpm_finetune.py --config_path conf/ljspeech_full.yaml

# 多卡 GPU 训练 (4卡)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train_voxcpm_finetune.py --config_path conf/ljspeech_full.yaml
```

## 步骤 4 — 监控训练过程

```bash
tensorboard --logdir checkpoints/ljspeech_full/logs
```

### 关键指标说明

| 指标 | 正常趋势描述 |
| :--- | :--- |
| `loss/diff` | 持续稳定下降，并在趋于收敛时变平缓 |
| `loss/stop` | 在前 100–200 步中迅速下降，随后保持在较低水平 |
| `grad_norm` | 大体保持在 0.3–2.0 范围内；偶尔出现尖峰是正常现象 |
| `lr` | 先进行余弦预热（Warm-up），随后逐渐衰减 |
| `val/loss` | 与训练集 loss 保持同步；**如果验证集 loss 上升而训练集 loss 仍在下降，请停止训练** |

### 何时停止训练

对于 TTS 微调，**1–2 个 epoch 通常就足够了**。表现最好的检查点通常不是最后一个。

- 建议设置配置文件中的 `valid_interval: 50` 和 `save_interval: 50`，以便有足够的回滚选择。
- 选择 `val/loss` 处于最低值时的检查点。
- 如果您没有准备验证集清单，可以用推理脚本在收敛区间内人工评测几个检查点，并选择听起来最好的一版。

> [!WARNING]
> 如果 `val/loss` 开始上升而 `train/loss` 持续下降，**请立即停止训练**并回滚检查点。这是 TTS 模型非常典型的过拟合特征：模型会开始忽略输入的文本，无论您输入什么内容，它都只会生成相同的语音模式。

---

## 步骤 5 — 进行推理测试

### 全参数微调 (SFT) 检查点推理

```bash
# 标准 TTS
python scripts/test_voxcpm_ft_infer.py \
    --ckpt_dir checkpoints/librispeech_full/latest \
    --text "She walked slowly along the quiet avenue, listening to the wind." \
    --output output_full.wav

# 声音克隆 (需传入参考音频和其对应的精确文本)
python scripts/test_voxcpm_ft_infer.py \
    --ckpt_dir checkpoints/librispeech_full/latest \
    --text "She walked slowly along the quiet avenue, listening to the wind." \
    --prompt_audio examples/reference_speaker.wav \
    --prompt_text  "Exact transcript of the reference audio." \
    --output output_full_cloned.wav
```

## 故障排查

### 显存溢出 (OOM)

LJSpeech 的音频片段时长跨度较大（2 秒至 35 秒不等）。`max_batch_tokens` 已经过滤了最长的一些音频，如果仍然发生 OOM，可以尝试：

```yaml
# 方案 1 — 调小 batch_size，但保持相同的有效 batch_size
batch_size:       8
grad_accum_steps: 2

# 方案 2 — 缩减 token 预算限制
max_batch_tokens: 4096
```

### Loss 不下降

- 验证清单文件 (Manifest) 中的音频路径是否正确，且所有文件是否均可读。
- LibriSpeech 的 FLAC 文件采样率为 16 kHz；配置中必须保持 `sample_rate: 16000` — 这与 AudioVAE 编码器的输入采样率一致，数据加载器（dataloader）会自动进行重采样。
- 检查文本标注是否已转换为首字母大写的句式，而不是全大写（ALL-CAPS）。

### 生成的音频忽略了输入文本

这是典型的过拟合现象。请回滚至更早的检查点：

```bash
ls checkpoints/ljspeech_full/   # 查找分叉/发散之前的步数

python scripts/test_voxcpm_ft_infer.py \
    --ckpt_dir checkpoints/ljspeech_full/step_0001000 \
    --text "Test sentence." \
    --output test.wav
```

为了防范此问题，在未来的训练中：一定要提供 `val_manifest`，使用 `valid_interval: 50`，并在 `val/loss` 掉头向上时及时停止训练。通常将训练控制在 1–3 个 epoch 内即可完美避免过拟合。
