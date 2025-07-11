import cv2
import os


def extract_video_segment(input_path, output_path, start_frame, end_frame):
    """
    截取视频从指定帧开始到指定帧结束的片段
    参数:
        input_path: 输入视频文件路径
        output_path: 输出视频文件路径
        start_frame: 起始帧号(从0开始计数)
        end_frame: 结束帧号(包含在内)
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入视频文件不存在: {input_path}")

    # 打开视频文件
    cap = cv2.VideoCapture(input_path)

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # # 验证帧范围有效性
    # if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
    #     raise ValueError(f"无效的帧范围: 总帧数={total_frames}, 起始帧={start_frame}, 结束帧={end_frame}")

    # 设置视频编码器 (MP4格式推荐使用H264编码)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 兼容性较好的MP4编码
    # 对于更新的系统，可以使用: fourcc = cv2.VideoWriter_fourcc(*'avc1')

    # 创建视频写入对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 进度计数器
    current_frame = start_frame
    processed_frames = 0
    total_to_process = end_frame - start_frame + 1

    print(f"开始处理: 总帧数 {total_to_process} (从 {start_frame} 到 {end_frame})")

    try:
        while current_frame <= end_frame:
            ret, frame = cap.read()

            if not ret:
                print(f"警告: 在帧 {current_frame} 读取失败")
                break

            # 写入输出视频
            out.write(frame)

            # 更新进度
            processed_frames += 1
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 每处理100帧显示一次进度
            if processed_frames % 100 == 0:
                progress = (processed_frames / total_to_process) * 100
                print(f"进度: {processed_frames}/{total_to_process} 帧 ({progress:.1f}%)")

    finally:
        # 确保资源被释放
        cap.release()
        out.release()
        print(f"处理完成! 已保存到: {output_path}")
        print(f"实际处理帧数: {processed_frames}/{total_to_process}")


# 使用示例
if __name__ == "__main__":
    input_video = "/home/jia/PycharmProjects/Deep-EIoU/tools/YOLOX_outputs/yolox_x_ch_sportsmot/track_vis/2025_07_08_13_10_32/1212.mp4"  # 输入视频文件
    output_video = "test.mp4"  # 输出视频文件
    start_frame = 1550  # 起始帧(0-indexed)
    end_frame = 1650  # 结束帧(包含在内)

    extract_video_segment(input_video, output_video, start_frame, end_frame)
