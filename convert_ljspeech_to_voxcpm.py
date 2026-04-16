#!/usr/bin/env python3
"""
LJSpeech to VoxCPM Dataset Converter

This script converts LJSpeech format datasets to the VoxCPM-compatible JSONL format.
LJSpeech format typically has two columns separated by "|" in a metadata file:
- First column: audio filename (with or without extension)
- Second column: transcription text

Usage:
    # Convert only
    python convert_ljspeech_to_voxcpm.py --metadata_paths /path/to/metadata.csv --output_path train.jsonl

    # Convert and split train/val
    python convert_ljspeech_to_voxcpm.py --metadata_paths /path/to/metadata.csv \\
        --output_path train.jsonl --val_size 300 --val_output_path val.jsonl
"""

import os
import random
import argparse
import json
from pathlib import Path


def parse_ljspeech_line(line):
    """
    Parse a single line from LJSpeech metadata file.
    
    Args:
        line (str): A line from the metadata file
        
    Returns:
        tuple: (audio_filename, text) or (None, None) if parsing fails
    """
    line = line.strip()
    if not line or line.startswith('#'):  # Skip empty lines and comments
        return None, None
    
    # Split by "|" - LJSpeech format
    parts = line.split('|')
    if len(parts) < 2:
        return None, None
    
    audio_filename = parts[0].strip()
    text = parts[1].strip()
    
    return audio_filename, text


def convert_ljspeech_to_jsonl(metadata_paths, output_path):
    """
    Convert LJSpeech format metadata to VoxCPM JSONL format.
    
    Args:
        metadata_paths (list): List of paths to LJSpeech metadata files
        output_path (str): Output JSONL file path
    """
    output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    skipped_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for metadata_path in metadata_paths:
            metadata_path = Path(metadata_path)
            
            if not metadata_path.exists():
                print(f"Warning: Metadata file not found, skipping: {metadata_path}")
                continue
            
            # Audio files are in the same directory as metadata file
            audio_dir = metadata_path.parent / "wavs"
            print(f"Processing dataset from: {metadata_path}")
            
            with open(metadata_path, 'r', encoding='utf-8') as meta_file:
                for line_num, line in enumerate(meta_file, 1):
                    audio_filename, text = parse_ljspeech_line(line)
                    
                    if audio_filename is None:
                        skipped_count += 1
                        continue
                    
                    # Construct full audio path
                    # audio_filename already includes extension from metadata
                    audio_path = audio_dir / audio_filename
                    
                    # Check if audio file exists
                    if not audio_path.exists():
                        print(f"Warning: Audio file not found: {audio_path}")
                        skipped_count += 1
                        continue
                    
                    # Create JSON object in VoxCPM format
                    json_obj = {
                        "audio": str(audio_path),
                        "text": text
                    }
                    
                    # Write to JSONL file
                    out_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                    converted_count += 1
                    
                    if converted_count % 1000 == 0:
                        print(f"Processed {converted_count} samples...")
    
    print(f"\nConversion complete!")
    print(f"Converted samples: {converted_count}")
    print(f"Skipped samples: {skipped_count}")
    print(f"Output saved to: {output_path}")

    return converted_count


def split_train_val(train_path, val_output_path, val_size, seed=42):
    """
    Randomly split a JSONL file into train and validation sets.

    Reads all records from `train_path`, shuffles them, writes the first
    `val_size` records to `val_output_path`, and overwrites `train_path`
    with the remaining records.

    Args:
        train_path (str | Path): Path to the full JSONL file (will be overwritten).
        val_output_path (str | Path): Path for the validation JSONL output.
        val_size (int): Number of samples to put in the validation set.
        seed (int): Random seed for reproducibility.
    """
    train_path = Path(train_path)
    val_output_path = Path(val_output_path)

    lines = train_path.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()]  # drop blank lines

    if val_size >= len(lines):
        raise ValueError(
            f"val_size ({val_size}) must be less than total samples ({len(lines)})"
        )

    random.seed(seed)
    random.shuffle(lines)

    val_lines = lines[:val_size]
    train_lines = lines[val_size:]

    val_output_path.parent.mkdir(parents=True, exist_ok=True)
    val_output_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")

    print(f"\nTrain/Val split complete!")
    print(f"  Train samples : {len(train_lines):,}  -> {train_path}")
    print(f"  Val   samples : {len(val_lines):,}  -> {val_output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert LJSpeech format dataset to VoxCPM format')
    parser.add_argument('--metadata_paths', nargs='+', required=True,
                        help='Paths to one or more LJSpeech metadata files (e.g., ds1/metadata.csv ds2/metadata.csv)')
    parser.add_argument('--output_path', default='train.jsonl',
                        help='Output JSONL file path (default: train.jsonl)')
    parser.add_argument('--val_size', type=int, default=300,
                        help='Number of samples to hold out as validation set (default: 0 = no split)')
    parser.add_argument('--val_output_path', default='val.jsonl',
                        help='Output path for the validation JSONL (default: val.jsonl)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/val split reproducibility (default: 42)')

    args = parser.parse_args()

    try:
        convert_ljspeech_to_jsonl(
            args.metadata_paths,
            args.output_path,
        )

        if args.val_size > 0:
            split_train_val(
                train_path=args.output_path,
                val_output_path=args.val_output_path,
                val_size=args.val_size,
                seed=args.seed,
            )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
