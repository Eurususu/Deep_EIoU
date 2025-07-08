import os
import re
from pathlib import Path
import natsort


def check_path_type(input_path):
    """
    判断输入路径是文件目录还是视频文件
    返回:
        "directory" - 目录
        "video" - 视频文件
        "invalid" - 无效路径
        "other" - 其他类型文件
    """
    # 1. 检查路径是否存在
    if not os.path.exists(input_path):
        return "invalid"

    # 2. 检查是否是目录
    if os.path.isdir(input_path):
        return "directory"

    # 3. 检查是否是文件
    if os.path.isfile(input_path):
        # 4. 通过扩展名初步判断
        video_extensions = {
            '.mp4', '.avi', '.mov', '.mkv', '.flv',
            '.wmv', '.webm', '.m4v', '.mpg', '.mpeg',
            '.3gp', '.ts', '.m2ts', '.vob','.h264'
        }

        file_extension = Path(input_path).suffix.lower()
        if file_extension in video_extensions:
            return "video"


def get_sorted_image_files(directory, recursive=False, natural_sort=True):
    """
    按文件名顺序获取目录中的图片文件

    参数:
    directory (str): 目标目录路径
    recursive (bool): 是否递归搜索子目录
    natural_sort (bool): 是否使用自然排序（数字顺序）

    返回:
    list: 排序后的图片文件路径列表
    """
    # 支持的图片扩展名
    image_extensions = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp',
        '.tiff', '.webp', '.svg', '.heic', '.jfif'
    }

    # 验证目录是否存在
    if not os.path.isdir(directory):
        raise ValueError(f"目录不存在: {directory}")

    # 收集图片文件路径
    image_files = []

    if recursive:
        # 递归搜索
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if Path(file_path).suffix.lower() in image_extensions:
                    image_files.append(file_path)
    else:
        # 非递归搜索
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and Path(file_path).suffix.lower() in image_extensions:
                image_files.append(file_path)

    # 排序逻辑
    if natural_sort:
        # 自然排序（数字顺序）
        try:
            # 使用natsort库（更高效）
            return natsort.natsorted(image_files)
        except ImportError:
            # 手动实现自然排序
            def natural_key(string):
                """生成自然排序键"""
                return [int(text) if text.isdigit() else text.lower()
                        for text in re.split(r'(\d+)', string)]

            return sorted(image_files, key=natural_key)
    else:
        # 普通字母数字排序
        return sorted(image_files)


def parse_detection_file(file_path):
    """
    解析检测结果文件并构建字典

    参数:
    file_path (str): 检测结果文件路径

    返回:
    dict: {frame_id: [[cx, cy, w, h, conf], ...]} 或空列表
    """
    # 读取文件内容
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 临时存储有检测框的帧数据
    frame_dict = {}

    # 记录最小和最大帧ID
    min_frame = float('inf')
    max_frame = float('-inf')

    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 7:
            continue

        # 解析帧ID（转换为整数）
        frame_id = int(float(parts[0]))

        # 更新帧范围
        min_frame = min(min_frame, frame_id)
        max_frame = max(max_frame, frame_id)

        # 解析检测框参数
        cx, cy, w, h = map(float, parts[2:6])
        conf = float(parts[6])

        # 添加到帧字典
        if frame_id not in frame_dict:
            frame_dict[frame_id] = []
        frame_dict[frame_id].append([cx, cy, w, h, conf])

    # 确保所有帧都有条目（包括空帧）
    result_dict = {}
    for frame_id in range(min_frame, max_frame + 1):
        result_dict[frame_id] = frame_dict.get(frame_id, [])

    return result_dict
