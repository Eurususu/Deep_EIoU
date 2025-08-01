from matplotlib.colors import hsv_to_rgb
from utils import *
import numpy as np


def generate_color_map(max_id):
    """生成ID到颜色的映射，使用HSV颜色空间确保颜色差异"""
    color_map = {}
    for obj_id in range(1, max_id + 1):
        # 使用HSV颜色空间，固定饱和度和亮度，变化色调
        hue = (obj_id * 0.618) % 1.0  # 黄金分割比例确保颜色分布均匀
        rgb = hsv_to_rgb([hue, 0.8, 0.9])
        # 转换为0-255范围的BGR格式
        color = tuple(int(c * 255) for c in rgb[::-1])
        color_map[obj_id] = color
    return color_map


def visualize_tracking(image_dir, output_dir, frame_data, color_map, fps=30):
    """可视化跟踪结果并保存为视频"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取视频尺寸（从第一张存在的图片）
    frame_size = None
    for frame_id in sorted(frame_data.keys()):
        img_name = f"{frame_id:06d}.jpg"
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                frame_size = (img.shape[1], img.shape[0])  # (width, height)
                break

    if frame_size is None:
        print("错误: 无法确定视频尺寸，没有找到有效的图片")
        return

    # 创建视频文件
    video_path = os.path.join(output_dir, "1212.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式编码器
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    if not video_writer.isOpened():
        print(f"错误: 无法创建视频文件 {video_path}")
        return

    print(f"开始创建视频: {video_path}, 尺寸: {frame_size}, FPS: {fps}")

    # 处理所有帧
    frame_count = 0
    skipped_frames = 0
    max_frame = max(frame_data.keys()) if frame_data else 0

    for frame_id in range(1, max_frame + 1):
        img_name = f"{frame_id:06d}.jpg"
        img_path = os.path.join(image_dir, img_name)

        # 读取图片
        img = None
        if os.path.exists(img_path):
            img = cv2.imread(img_path)

        # 如果图片不存在或读取失败，创建空白帧
        if img is None:
            img = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            skipped_frames += 1
            # 在空白帧上显示警告
            cv2.putText(img, f"Frame {frame_id} missing",
                        (50, frame_size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 绘制当前帧的检测框
        boxes = frame_data.get(frame_id, [])
        for (obj_id, x, y, w, h) in boxes:
            color = color_map.get(obj_id, (0, 0, 255))  # 默认红色
            # 绘制矩形
            cv2.rectangle(img,
                          (int(x), int(y)),
                          (int(x + w), int(y + h)),
                          color, 2)
            # 绘制ID标签
            label = f"ID:{obj_id}"
            cv2.putText(img, label,
                        (int(x), int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 在帧左上角显示帧号
        cv2.putText(img, f"Frame: {frame_id}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 写入视频
        video_writer.write(img)
        frame_count += 1

        # 每处理100帧打印进度
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧...")

    # 释放视频写入器
    video_writer.release()

    print(f"\n视频创建完成! 总帧数: {frame_count}, 缺失帧数: {skipped_frames}")
    print(f"视频保存路径: {video_path}")


def visualize_track_id(track_id, data_file, image_dir, output_video_path, fps=30, trail_length=50):
    """
    可视化指定轨迹ID的边界框和运动轨迹

    参数:
    track_id (int): 要可视化的轨迹ID
    data_file (str): 跟踪数据文件路径
    image_dir (str): 原始图片目录
    output_video_path (str): 输出视频路径
    fps (int): 视频帧率
    trail_length (int): 轨迹长度（保留的历史帧数）
    """
    # 步骤1: 解析跟踪数据文件
    track_data = defaultdict(list)
    min_frame = float('inf')
    max_frame = float('-inf')

    with open(data_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue

            frame_id = int(float(parts[0]))
            obj_id = int(float(parts[1]))

            # 更新帧范围
            min_frame = min(min_frame, frame_id)
            max_frame = max(max_frame, frame_id)

            # 只处理指定轨迹ID
            if obj_id == track_id:
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                cx = x + w / 2  # 中心点x坐标
                cy = y + h / 2  # 中心点y坐标

                track_data[frame_id] = [x, y, w, h, cx, cy]

    if not track_data:
        print(f"未找到轨迹ID {track_id} 的数据")
        return

    print(f"找到 {len(track_data)} 帧包含轨迹ID {track_id}")

    # 步骤2: 获取视频尺寸
    frame_size = None
    for frame_id in range(min_frame, max_frame + 1):
        img_path = os.path.join(image_dir, f"{frame_id:06d}.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                frame_size = (img.shape[1], img.shape[0])
                break

    if frame_size is None:
        print("无法确定视频尺寸，没有找到有效的图片")
        return

    # 步骤3: 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    if not video_writer.isOpened():
        print(f"无法创建视频文件 {output_video_path}")
        return

    print(f"开始创建视频: {output_video_path}, 尺寸: {frame_size}, FPS: {fps}")

    # 存储轨迹点历史
    trail_points = []

    # 步骤4: 处理每一帧
    for frame_id in range(min_frame, max_frame + 1):
        img_path = os.path.join(image_dir, f"{frame_id:06d}.jpg")
        img = None

        # 读取图片
        if os.path.exists(img_path):
            img = cv2.imread(img_path)

        # 如果图片不存在或读取失败，创建空白帧
        if img is None:
            img = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            cv2.putText(img, f"Frame {frame_id} missing",
                        (50, frame_size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 检查当前帧是否有指定轨迹ID的数据
        if frame_id in track_data:
            x, y, w, h, cx, cy = track_data[frame_id]

            # 绘制边界框
            cv2.rectangle(img,
                          (int(x), int(y)),
                          (int(x + w), int(y + h)),
                          (0, 255, 0), 2)  # 绿色边界框

            # 绘制ID标签
            label = f"ID: {track_id}"
            cv2.putText(img, label,
                        (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 添加当前中心点到轨迹历史
            trail_points.append((int(cx), int(cy)))

            # 限制轨迹长度
            if len(trail_points) > trail_length:
                trail_points.pop(0)

        # 绘制轨迹（如果至少有两个点）
        if len(trail_points) >= 2:
            # 绘制轨迹线
            for i in range(1, len(trail_points)):
                # 根据轨迹点的新旧程度设置颜色（越新越亮）
                color_intensity = int(255 * i / len(trail_points))
                color = (0, 0, 255 - color_intensity)  # 从蓝色到红色渐变

                cv2.line(img, trail_points[i - 1], trail_points[i], color, 2)

            # 绘制轨迹起点和终点
            cv2.circle(img, trail_points[0], 5, (255, 0, 0), -1)  # 起点（蓝色）
            cv2.circle(img, trail_points[-1], 5, (0, 0, 255), -1)  # 终点（红色）

        # 添加帧信息
        cv2.putText(img, f"Frame: {frame_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(img, f"Track ID: {track_id}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(img, f"Trail Length: {len(trail_points)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 写入视频
        video_writer.write(img)

        # 每处理100帧打印进度
        if frame_id % 100 == 0:
            print(f"已处理 {frame_id} 帧...")

    # 释放视频写入器
    video_writer.release()
    print(f"\n视频创建完成! 保存路径: {output_video_path}")


if __name__ == "__main__":
    image_dir = "/home/jia/PycharmProjects/gta-link/test_data/1212/img1"  # 图片目录路径
    output_dir = "/home/jia/PycharmProjects/Deep-EIoU/vis_result"  # 输出目录路径
    data_file = "/home/jia/PycharmProjects/gta-link/test_data/DeepEIoU_SoccerNet_Split+Connect/1212.txt"  # MOT数据文件路径

    # 读取数据
    with open(data_file, 'r') as f:
        data_lines = f.readlines()

    # 解析数据
    frame_data, id_count = parse_mot_data(data_lines)

    if not frame_data:
        print("未解析到有效数据")


    # 生成颜色映射
    max_id = max(id_count.keys()) if id_count else 0
    color_map = generate_color_map(max_id)

    # 可视化跟踪结果
    visualize_tracking(image_dir, output_dir, frame_data, color_map)

    # visualize_track_id(
    #     track_id=7,
    #     data_file=data_file,
    #     image_dir=image_dir,
    #     output_video_path="vis_result/tracking_id7.mp4",
    #     fps=30,
    #     trail_length=100  # 保留100帧的轨迹
    # )

    print("\n处理完成! 结果保存在:", output_dir)