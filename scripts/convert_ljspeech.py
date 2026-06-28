import argparse
import json
import os
import csv

def convert_ljspeech_to_jsonl(ljspeech_file, output_file, wavs_dir=None):
    # wavs_dir 未指定时，自动推断为 metadata 文件同级目录下的 wavs/ 子文件夹
    # 这与 LJSpeech 的标准目录结构保持一致：
    #   LJSpeech-1.1/
    #     ├── metadata.csv
    #     └── wavs/
    if wavs_dir is None:
        wavs_dir = os.path.join(os.path.dirname(os.path.abspath(ljspeech_file)), "wavs")
        print(f"未指定 --wavs_dir，自动使用: {wavs_dir}")

    count = 0
    with open(ljspeech_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        # 使用 csv 模块来解析标准的 csv 文件，默认处理逗号分隔
        # 如果仍有特殊情况，用户可以在此调整 delimiter (比如 delimiter='|')
        reader = csv.reader(infile, delimiter='|')
        for row in reader:
            if not row:
                continue
            
            if len(row) >= 2:
                filename = row[0].strip()
                text = row[1].strip()
                
                audio_path = os.path.join(wavs_dir, filename).replace("\\", "/")
                
                json_obj = {
                    "text": text,
                    "audio": audio_path
                }
                
                outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                count += 1
            else:
                print(f"跳过格式不合法的行: {row}")
                
    print(f"转换完成！共成功转换 {count} 条数据。")
    print(f"JSONL 文件已保存至: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将双栏 LJSpeech 格式转换为微调专用的 JSONL 格式")
    parser.add_argument("--input", "-i", required=True, help="输入的 LJSpeech 标注文件路径 (如 metadata.csv)")
    parser.add_argument("--wavs_dir", "-w", default=None, help="存放音频文件的目录 (将与第一栏的文件名拼接)。省略时自动使用 metadata 文件同级的 wavs/ 子目录")
    parser.add_argument("--output", "-o", default="train.jsonl", help="输出的 JSONL 文件路径 (默认: train.jsonl)")
    
    args = parser.parse_args()
    
    convert_ljspeech_to_jsonl(args.input, args.output, args.wavs_dir)
