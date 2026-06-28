import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from huggingface_hub import snapshot_download

# VoxCPM imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from voxcpm.model.voxcpm import VoxCPMModel
from voxcpm.model.voxcpm2 import VoxCPM2Model
from voxcpm.training.packers import AudioFeatureProcessingPacker


def main():
    parser = argparse.ArgumentParser(description="Pre-extract VAE latent features for offline training.")
    parser.add_argument("--input_manifest", "-i", required=True, help="Path to input train.jsonl (with audio paths).")
    parser.add_argument("--output_manifest", "-o", required=True, help="Path to output latents.jsonl.")
    parser.add_argument("--latent_dir", "-d", required=True, help="Directory to save the extracted .pt latent files.")
    parser.add_argument("--pretrained_path", "-p", required=True, help="Path or HuggingFace ID to the pretrained VoxCPM model (for AudioVAE).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for VAE extraction.")
    
    args = parser.parse_args()

    # 1. Load AudioVAE
    pretrained_path = args.pretrained_path
    if not os.path.isdir(pretrained_path):
        print(f"Downloading from HuggingFace Hub: {pretrained_path} ...")
        pretrained_path = snapshot_download(repo_id=pretrained_path)
    
    with open(os.path.join(pretrained_path, "config.json"), "r", encoding="utf-8") as f:
        arch = json.load(f).get("architecture", "voxcpm").lower()
    
    model_cls = VoxCPM2Model if arch == "voxcpm2" else VoxCPMModel
    print(f"Loading {model_cls.__name__} from {pretrained_path} to extract AudioVAE...")
    
    # We only need the base model structure to extract the VAE.
    base_model = model_cls.from_local(pretrained_path, optimize=False, training=False)
    audio_vae = base_model.audio_vae.to(args.device).eval()
    
    # Initialize a dummy packer just to reuse its extract_audio_feats method
    packer = AudioFeatureProcessingPacker(
        dataset_cnt=1, 
        max_len=8192, 
        patch_size=base_model.config.patch_size, 
        feat_dim=base_model.config.feat_dim, 
        audio_vae=audio_vae
    )

    expected_sr = audio_vae.sample_rate
    print(f"AudioVAE expects sample rate: {expected_sr} Hz")
    
    # 2. Process data
    latent_dir = Path(args.latent_dir)
    latent_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(args.input_manifest, 'r', encoding='utf-8') as f_in, \
         open(args.output_manifest, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        for i, line in enumerate(tqdm(lines, desc="Extracting latents")):
            if not line.strip(): continue
            item = json.loads(line)
            audio_path = item.get("audio")
            if not audio_path or not os.path.isfile(audio_path):
                print(f"Warning: Audio file not found for item {i}: {audio_path}")
                continue
                
            # Load and resample audio
            try:
                wav, sr = torchaudio.load(audio_path, backend="soundfile")
                if wav.size(0) > 1:
                    wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
                if sr != expected_sr:
                    resampler = torchaudio.transforms.Resample(sr, expected_sr)
                    wav = resampler(wav)
            except Exception as e:
                print(f"Failed to load audio {audio_path}: {e}")
                continue
                
            wav = wav.squeeze(0).to(args.device) # Shape: [T_wav]
            
            # Extract features using Packer's native method
            with torch.no_grad():
                audio_feats, audio_duration = packer.extract_audio_feats(wav)
                
            # audio_feats shape from packer is [B, T, P, C]. Since B=1, we squeeze it.
            audio_feats = audio_feats.squeeze(0).cpu()
            
            # Save latent
            filename = f"latent_{i}_{Path(audio_path).stem}.pt"
            latent_path = latent_dir / filename
            torch.save(audio_feats, latent_path)
            
            # Update manifest item
            item["audio"] = str(latent_path.as_posix())
            item["is_latent"] = True
            item["duration"] = audio_duration
            
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
            
    print(f"Extraction complete! Successfully processed {count} files.")
    print(f"New manifest saved to: {args.output_manifest}")

if __name__ == "__main__":
    main()
