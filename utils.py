import cv2
import os
from pathlib import Path
from collections import defaultdict
import numpy as np


def extract_frames(video_path: str, output_dir: str, start_frame=0, end_frame=None, frame_step=1):
    """
    提取视频帧并保存为序列图片

    参数:
    video_path: 输入视频文件路径
    output_dir: 输出图片保存目录
    start_frame: 起始帧号 (默认为0)
    end_frame: 结束帧号 (默认为视频最后一帧)
    frame_step: 帧间隔 (默认为1，即所有帧)
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频基本信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {total_frames}帧, {fps:.2f}FPS, 分辨率: {width}x{height}")

    # 设置结束帧（如果未指定）
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    # 设置起始位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    saved_count = 0
    skipped_count = 0

    print(f"开始提取帧: {start_frame} 到 {end_frame}, 步长 {frame_step}...")

    while frame_count <= end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        # 按步长保存帧
        if (frame_count - start_frame) % frame_step == 0:
            # 生成6位数字文件名
            filename = f"{saved_count + 1:06d}.jpg"
            output_path = os.path.join(output_dir, filename)

            # 保存为JPG（质量95%）
            cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_count += 1
        else:
            skipped_count += 1

        frame_count += 1

        # 每处理100帧显示一次进度
        if frame_count % 100 == 0:
            print(f"已处理: {frame_count}/{end_frame} 帧, 已保存: {saved_count} 张图片")

    cap.release()
    print(f"完成! 共处理 {frame_count} 帧, 保存 {saved_count} 张图片, 跳过 {skipped_count} 帧")
    return saved_count


def parse_mot_data(data_lines):
    """
    解析MOT数据格式
    返回:
        dict: {frame_id: [(obj_id, cx, cy, w, h), ...]}
        dict: {obj_id: count}
    """
    frame_data = defaultdict(list)
    id_count = defaultdict(int)

    for line in data_lines:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue

        frame_id = int(float(parts[0]))  # 1.0 -> 1
        obj_id = int(parts[1])
        x, y, w, h = map(float, parts[2:6])



        frame_data[frame_id].append((obj_id, x, y, w, h))
        id_count[obj_id] += 1

    return frame_data, id_count


# jjh
def convert_tlwh_to_xyxy(bboxes_tlwh):
    """
    将[x_tl, y_tl, w, h]格式的边界框转换为[x1, y1, x2, y2]格式

    参数:
        bboxes_tlwh: numpy数组，形状为(N, 4)或(4,)，包含多个边界框

    返回:
        bboxes_xyxy: 转换后的边界框数组，形状与输入相同
    """
    # 确保输入是二维数组 (N, 4)
    if bboxes_tlwh.ndim == 1:
        bboxes_tlwh = bboxes_tlwh[np.newaxis, :]

    # 提取坐标和尺寸
    x_tl = bboxes_tlwh[:, 0]
    y_tl = bboxes_tlwh[:, 1]
    w = bboxes_tlwh[:, 2]
    h = bboxes_tlwh[:, 3]

    # 计算右下角坐标
    x_br = x_tl + w
    y_br = y_tl + h

    # 组合成xyxy格式
    bboxes_xyxy = np.column_stack((x_tl, y_tl, x_br, y_br))

    return bboxes_xyxy


# if __name__ == "__main__":
    # # 使用示例
    # video_file = "/home/jia/PycharmProjects/gta-link/test_data/1212.mp4"  # 替换为你的视频路径
    # output_folder = "/home/jia/PycharmProjects/gta-link/test_data/1212/img1"  # 输出目录
    #
    # # 提取所有帧
    # extract_frames(video_file, output_folder)

    # 可选：提取部分帧
    # extract_frames(video_file, output_folder,
    #                start_frame=100,  # 从第100帧开始
    #                end_frame=500,    # 到第500帧结束
    #                frame_step=2)     # 每隔1帧保存1帧
