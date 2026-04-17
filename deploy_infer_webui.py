import os
import sys
import json
import gradio as gr
import torch
from pathlib import Path
from typing import Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add src to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Default pretrained model path: prefer VoxCPM2 if it exists, fallback to openbmb/VoxCPM2
_v2_path = project_root / "models" / "VoxCPM2"
default_pretrained_path = str(_v2_path) if _v2_path.exists() else "openbmb/VoxCPM2"

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig
import numpy as np

# Global variables
current_model: Optional[VoxCPM] = None

def scan_lora_checkpoints(root_dir="lora", with_info=False):
    """
    Scans for LoRA checkpoints in the lora directory.
    """
    checkpoints = []
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)

    for root, dirs, files in os.walk(root_dir):
        if "lora_weights.safetensors" in files:
            rel_path = os.path.relpath(root, root_dir)
            if with_info:
                base_model = None
                lora_config_file = os.path.join(root, "lora_config.json")
                if os.path.exists(lora_config_file):
                    try:
                        with open(lora_config_file, "r", encoding="utf-8") as f:
                            lora_info = json.load(f)
                        base_model = lora_info.get("base_model", "Unknown")
                    except (json.JSONDecodeError, OSError):
                        pass
                checkpoints.append((rel_path, base_model))
            else:
                checkpoints.append(rel_path)

    return sorted(checkpoints, reverse=True)


def load_lora_config_from_checkpoint(lora_path):
    """Load LoRA config from lora_config.json if available."""
    lora_config_file = os.path.join(lora_path, "lora_config.json")
    if os.path.exists(lora_config_file):
        try:
            with open(lora_config_file, "r", encoding="utf-8") as f:
                lora_info = json.load(f)
            lora_cfg_dict = lora_info.get("lora_config", {})
            if lora_cfg_dict:
                return LoRAConfig(**lora_cfg_dict), lora_info.get("base_model")
        except Exception as e:
            print(f"Warning: Failed to load lora_config.json: {e}", file=sys.stderr)
    return None, None


def get_default_lora_config():
    """Return default LoRA config for hot-swapping support."""
    return LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        r=32,
        alpha=16,
        target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
        target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def load_model(pretrained_path, lora_path=None):
    global current_model
    print(f"Loading model from {pretrained_path}...", file=sys.stderr)

    lora_config = None
    lora_weights_path = None

    if lora_path:
        full_lora_path = os.path.join("lora", lora_path)
        if os.path.exists(full_lora_path):
            lora_weights_path = full_lora_path
            lora_config, _ = load_lora_config_from_checkpoint(full_lora_path)
            if lora_config:
                print(f"Loaded LoRA config from {full_lora_path}/lora_config.json", file=sys.stderr)
            else:
                lora_config = get_default_lora_config()
                print("Using default LoRA config (lora_config.json not found)", file=sys.stderr)

    if lora_config is None:
        lora_config = get_default_lora_config()

    current_model = VoxCPM.from_pretrained(
        hf_model_id=pretrained_path,
        load_denoiser=False,
        optimize=False,
        lora_config=lora_config,
        lora_weights_path=lora_weights_path,
    )
    return "Model loaded successfully!"


def run_inference(text, ref_wav, lora_selection, cfg_scale, steps, seed, pretrained_path=None):
    if current_model is None:
        base_model_path = pretrained_path if pretrained_path and pretrained_path.strip() else default_pretrained_path

        if lora_selection and lora_selection != "None":
            full_lora_path = os.path.join("lora", lora_selection)
            lora_config_file = os.path.join(full_lora_path, "lora_config.json")
            if os.path.exists(lora_config_file):
                try:
                    with open(lora_config_file, "r", encoding="utf-8") as f:
                        lora_info = json.load(f)
                    saved_base_model = lora_info.get("base_model")

                    if saved_base_model and os.path.exists(saved_base_model):
                        base_model_path = saved_base_model
                        print(f"Using base model from LoRA config: {base_model_path}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Failed to read base_model from LoRA config: {e}", file=sys.stderr)

        try:
            print(f"Loading base model: {base_model_path}", file=sys.stderr)
            lora_path = lora_selection if lora_selection and lora_selection != "None" else None
            load_model(base_model_path, lora_path=lora_path)
            if lora_selection and lora_selection != "None":
                print(f"Model loaded for LoRA: {lora_selection}", file=sys.stderr)
        except Exception as e:
            error_msg = f"Failed to load model from {base_model_path}: {str(e)}"
            print(error_msg, file=sys.stderr)
            return None, error_msg

    assert current_model is not None, "Model must be loaded before inference"
    if lora_selection and lora_selection != "None":
        full_lora_path = os.path.join("lora", lora_selection)
        print(f"Hot-loading LoRA: {full_lora_path}", file=sys.stderr)
        try:
            current_model.load_lora(full_lora_path)
            current_model.set_lora_enabled(True)
        except Exception as e:
            print(f"Error loading LoRA: {e}", file=sys.stderr)
            return None, f"Error loading LoRA: {e}"
    else:
        print("Disabling LoRA", file=sys.stderr)
        current_model.set_lora_enabled(False)

    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)

    final_ref_wav = None
    if ref_wav and ref_wav.strip():
        final_ref_wav = ref_wav

    try:
        audio_np = current_model.generate(
            text=text,
            reference_wav_path=final_ref_wav, # Using Voice Cloning Mode
            cfg_value=cfg_scale,
            inference_timesteps=steps,
            denoise=False,
        )
        return (current_model.tts_model.sample_rate, audio_np), "Generation Success"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

# --- GUI Layout ---

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.title-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 8px;
    padding: 15px 25px;
    margin-bottom: 15px;
    border: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.title-section h1, .title-section h3, .title-section p {
    color: white;
    text-shadow: none;
    margin: 0;
}
.title-section h1 { font-size: 28px; font-weight: 600; line-height: 1.2; }
.title-section h3 { font-size: 14px; font-weight: 400; margin-top: 5px; opacity: 0.9; }
.tabs {
    background: white;
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 12px;
    padding: 12px 30px;
    font-weight: 600;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}
.button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
}
.button-refresh {
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    border: none;
    border-radius: 10px;
    padding: 8px 20px;
    font-weight: 500;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 2px 10px rgba(132, 250, 176, 0.3);
}
.button-refresh:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(132, 250, 176, 0.4);
}
.form-section {
    background: white;
    border-radius: 20px;
    padding: 30px;
    margin: 15px 0;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.05);
}
.input-field {
    border-radius: 12px;
    border: 2px solid #e0e0e0;
    padding: 12px 16px;
    transition: all 0.3s ease;
    background: #fafafa;
}
.input-field:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    background: white;
}
"""

with gr.Blocks(title="VoxCPM Inference WebUI", theme=gr.themes.Soft(), css=custom_css) as app:

    with gr.Tabs(elem_classes="tabs"):
        with gr.Tab("🎵 语音克隆与合成"):
            with gr.Row():
                with gr.Column(scale=35, elem_classes="form-section"):
                    gr.Markdown("#### 📝 输入配置")

                    infer_text = gr.TextArea(
                        label="💬 合成文本",
                        value="Hello, this is a test of voice cloning using VoxCPM.",
                        elem_classes="input-field",
                        lines=4,
                        placeholder="输入要合成的文本内容...",
                    )

                    gr.Markdown("**🎭 声音克隆参考（必填以启用克隆）**")
                    
                    ref_wav = gr.Audio(label="🎵 目标发音人参考音频", type="filepath", elem_classes="input-field")
                    gr.Markdown("*注意：无需输入参考文本，模型会自动进行端到端音色和情绪克隆。*")

                with gr.Column(scale=35, elem_classes="form-section"):
                    gr.Markdown("#### 🤖 LoRA 模型选择")

                    lora_select = gr.Dropdown(
                        label="🎯 使用 LoRA 权重 (可选)",
                        choices=["None"] + scan_lora_checkpoints(),
                        value="None",
                        interactive=True,
                        elem_classes="input-field",
                        info="选择训练好的 LoRA 风格模型",
                    )

                    refresh_lora_btn = gr.Button("🔄 刷新模型列表", elem_classes="button-refresh", size="sm")

                    gr.Markdown("#### ⚙️ 生成参数")

                    cfg_scale = gr.Slider(
                        label="🎛️ CFG Scale",
                        minimum=1.0, maximum=5.0, value=2.0, step=0.1,
                    )

                    steps = gr.Slider(
                        label="🔢 推理步数",
                        minimum=1, maximum=50, value=10, step=1,
                    )

                    seed = gr.Number(
                        label="🎲 随机种子 (-1 为随机)",
                        value=-1, precision=0, elem_classes="input-field",
                    )

                    generate_btn = gr.Button("🎵 开始生成", variant="primary", elem_classes="button-primary", size="lg")

                with gr.Column(scale=30, elem_classes="form-section"):
                    gr.Markdown("#### 🎧 生成结果")

                    audio_out = gr.Audio(label="", elem_classes="input-field", show_label=False)

                    gr.Markdown("#### 📋 状态信息")

                    status_out = gr.Textbox(
                        label="", interactive=False, elem_classes="input-field",
                        show_label=False, lines=3, placeholder="等待生成...",
                    )
                    
            def refresh_loras():
                choices = ["None"] + [ckpt[0] for ckpt in scan_lora_checkpoints(with_info=True)]
                return gr.update(choices=choices, value="None")

            refresh_lora_btn.click(refresh_loras, outputs=[lora_select])

            generate_btn.click(
                run_inference,
                inputs=[
                    infer_text,
                    ref_wav,
                    lora_select,
                    cfg_scale,
                    steps,
                    seed,
                ],
                outputs=[audio_out, status_out],
            )

if __name__ == "__main__":
    os.makedirs("lora", exist_ok=True)
    app.queue().launch(server_name="0.0.0.0", server_port=7860)
