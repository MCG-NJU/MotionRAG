import os
import subprocess
import argparse
from tqdm.contrib.concurrent import process_map


def process_video(args):
    file, input_dir, output_dir = args
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, file)
    cmd = [
        'ffmpeg', '-hide_banner',
        '-i', input_path,
        '-vf', 'scale=-2:240',  # 修正scale语法
        '-c:v', 'libx264', '-crf', '20',
        '-preset', 'medium', '-c:a', 'copy',
        '-y',
        output_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"处理文件 {file} 时出错：")
            print(result.stderr)
    except Exception as e:
        print(f"处理文件 {file} 时发生异常：{str(e)}")


def main():
    parser = argparse.ArgumentParser(description="批量处理MP4视频，短边缩小到240像素，保留音频。")
    parser.add_argument('-i', '--input_dir', type=str, help='输入目录路径')
    parser.add_argument('-o', '--output_dir', type=str, help='输出目录路径')
    parser.add_argument('-p', '--processes', type=int, default=64, help='并行进程数（默认64）')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mp4_files = [f for f in os.listdir(args.input_dir) if f.endswith('.mp4')]

    if not mp4_files:
        print(f"在 {args.input_dir} 中没有找到MP4文件")
        return

    print(f"找到 {len(mp4_files)} 个MP4文件")

    # 准备参数
    process_args = [(file, args.input_dir, args.output_dir) for file in mp4_files]

    # 使用 process_map
    process_map(process_video, process_args, max_workers=args.processes, chunksize=4)

    print("所有视频处理完成！")


if __name__ == "__main__":
    main()
