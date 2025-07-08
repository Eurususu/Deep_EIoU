import os
import numpy as np
import torch
import sys
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


sys.path.append('.')


from reid.torchreid.utils import FeatureExtractor
from yolox.exp import get_exp
from yolox.utils import fuse_model,  postprocess
from yolox.data.data_augment import preproc
from model import Net



class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)[
            'net_dict']
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.size = (64, 128)   #（128,128）
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch, featuremap=True)
        return features


def visualize_feature_maps(img, det, cropped_imgs, feature_map_np, save_path=None):
    """
    可视化特征热力图并叠加到原始图像上
    Args:
        img: 原始图像 (BGR格式)
        det: 检测结果数组 [x1, y1, x2, y2, conf, cls]
        cropped_imgs: 裁剪后的图像列表 (BGR格式)
        feature_map_np: 特征图数组 (N, C, H', W')
        save_path: 结果保存路径
    Returns:
        superimposed_img: 叠加热力图的原始图像 (RGB格式)
    """
    # 创建原始图像的副本并转换为RGB
    img_with_heatmap = img.copy()
    img_with_heatmap_rgb = cv2.cvtColor(img_with_heatmap, cv2.COLOR_BGR2RGB)

    # 用于保存每个检测框的可视化结果
    crop_visualizations = []

    for i, (crop_img, feat_map) in enumerate(zip(cropped_imgs, feature_map_np)):
        # 获取当前检测框坐标
        x1, y1, x2, y2 = map(int, det[i, :4])

        # 计算特征图热力图 - 使用通道范数
        heatmap = np.linalg.norm(feat_map, axis=0)  # (H', W')

        # 归一化
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # 将热力图缩放到裁剪图像尺寸
        heatmap_resized = cv2.resize(heatmap, (crop_img.shape[1], crop_img.shape[0]))

        # 转换为彩色热力图 (BGR格式)
        heatmap_colored_bgr = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # 修复点1：将热力图转换为RGB
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

        # 叠加到裁剪图像上 (确保都是RGB)
        crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        superimposed_crop = cv2.addWeighted(crop_img_rgb, 0.5, heatmap_colored_rgb, 0.5, 0)
        crop_visualizations.append(superimposed_crop)

        # 将热力图叠加到原始图像对应位置
        crop_region = img_with_heatmap_rgb[y1:y2, x1:x2]
        if crop_region.size > 0:
            # 确保尺寸匹配
            if crop_region.shape[:2] == heatmap_colored_rgb.shape[:2]:
                superimposed_region = cv2.addWeighted(crop_region, 0.5, heatmap_colored_rgb, 0.5, 0)
                img_with_heatmap_rgb[y1:y2, x1:x2] = superimposed_region

        # 在原始图像上绘制检测框 (在BGR图像上绘制)
        cv2.rectangle(img_with_heatmap, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_with_heatmap, f"ID: {i}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 创建完整可视化
    fig = plt.figure(figsize=(20, 12))

    # 原始图像+热力图 (已经是RGB)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img_with_heatmap_rgb)
    ax1.set_title("Original Image with Attention Heatmaps")
    ax1.axis('off')

    # 原始检测结果 (需要转换为RGB)
    ax2 = fig.add_subplot(2, 2, 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax2.imshow(img_rgb)
    for i, d in enumerate(det):
        x1, y1, x2, y2 = map(int, d[:4])
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='green', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x1, y1 - 10, f'ID: {i}', color='green', fontsize=10)
    ax2.set_title("Detection Results")
    ax2.axis('off')

    # 裁剪区域热力图 (已经是RGB)
    if crop_visualizations:
        ax3 = fig.add_subplot(2, 2, 3)
        # 创建网格显示所有裁剪区域
        cols = min(4, len(crop_visualizations))
        rows = (len(crop_visualizations) + cols - 1) // cols

        # 创建空白画布
        max_h = max(cv.shape[0] for cv in crop_visualizations)
        max_w = max(cv.shape[1] for cv in crop_visualizations)
        grid_img = np.zeros((max_h * rows, max_w * cols, 3), dtype=np.uint8)

        # 填充网格
        for idx, crop in enumerate(crop_visualizations):
            row_idx = idx // cols
            col_idx = idx % cols
            h, w = crop.shape[:2]
            grid_img[row_idx * max_h:row_idx * max_h + h, col_idx * max_w:col_idx * max_w + w] = crop

        ax3.imshow(grid_img)
        ax3.set_title(f"Cropped Regions with Attention ({len(crop_visualizations)} detected)")
        ax3.axis('off')

    # 特征图通道可视化
    if feature_map_np.size > 0:
        ax4 = fig.add_subplot(2, 2, 4)
        # 随机选择16个通道
        num_channels = min(16, feature_map_np.shape[1])
        selected_channels = np.random.choice(feature_map_np.shape[1], num_channels, replace=False)

        # 创建通道网格
        channel_grid = []
        for c in selected_channels:
            channel_img = feature_map_np[0, c]  # 第一个检测框的特征
            # 归一化
            channel_img = (channel_img - channel_img.min()) / (channel_img.max() - channel_img.min() + 1e-8)
            channel_img = (channel_img * 255).astype(np.uint8)

            # 修复点2：生成RGB格式的热力图
            channel_img_bgr = cv2.applyColorMap(channel_img, cv2.COLORMAP_VIRIDIS)
            channel_img_rgb = cv2.cvtColor(channel_img_bgr, cv2.COLOR_BGR2RGB)
            channel_grid.append(channel_img_rgb)

        # 合并通道图像
        cols = 4
        rows = (num_channels + cols - 1) // cols
        max_ch_h, max_ch_w = channel_grid[0].shape[:2]
        channels_img = np.zeros((max_ch_h * rows, max_ch_w * cols, 3), dtype=np.uint8)

        for idx, ch_img in enumerate(channel_grid):
            r = idx // cols
            c = idx % cols
            h, w = ch_img.shape[:2]
            channels_img[r * max_ch_h:r * max_ch_h + h, c * max_ch_w:c * max_ch_w + w] = ch_img
            # 添加通道编号
            cv2.putText(channels_img, f"Ch{selected_channels[idx]}",
                        (c * max_ch_w + 5, r * max_ch_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ax4.imshow(channels_img)
        ax4.set_title("Feature Map Channels (Random Selection)")
        ax4.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")

    plt.close()

    return img_with_heatmap_rgb


def get_det_from_text(text_file, current_frame):
    """
    从文本文件中读取指定帧的检测框信息
    Args:
        text_file: 检测结果文本文件路径
        current_frame: 当前处理的帧号（注意：文本文件中的帧号从1开始）
    Returns:
        det: 检测结果数组，格式为 [x1, y1, x2, y2, conf, track_id, 0]
    """
    # 确保当前帧号在文本文件范围内
    current_frame = int(current_frame)

    # 读取文本文件所有行
    if not os.path.exists(text_file):
        print(f"Error: Detection file {text_file} not found!")
        return np.empty((0, 7))

    with open(text_file, 'r') as f:
        lines = f.readlines()

    # 解析每行数据
    detections = []
    for line in lines:
        # 移除换行符并分割字段
        parts = line.strip().split(',')

        # 确保有足够字段
        if len(parts) < 7:
            continue

        try:
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])

            # 只处理当前帧的检测结果
            if frame_id == current_frame:
                # 计算x2, y2
                x2 = x1 + w
                y2 = y1 + h

                # 创建检测结果数组 [x1, y1, x2, y2, conf, track_id, 0]
                det = [x1, y1, x2, y2, conf, track_id, 0]
                detections.append(det)

        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line.strip()} - {str(e)}")
            continue

    # 转换为numpy数组
    if detections:
        det_array = np.array(detections, dtype=np.float32)
    else:
        det_array = np.empty((0, 7))

    return det_array


if __name__ == "__main__":
    device = 'cuda'
    ckpt_file = "checkpoints/best_ckpt.pth.tar"
    model_name = "osnet_x1_0"
    model_path = "checkpoints/sports_model.pth.tar-60"
    imgs_dir = "/home/jia/PycharmProjects/gta-link/test_data/1212/img1"
    output_dir = "/home/jia/PycharmProjects/Deep-EIoU/Deep-EIoU/feature_vis"
    fuse = False
    half = False
    img_size = [800, 2880]
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    start_frame = 8120
    end_frame = 8147 # 8147
    text_file = "/home/jia/PycharmProjects/Deep-EIoU/filtered_results.txt"


    exp = get_exp("yolox/yolox_x_ch_sportsmot.py", None)
    model = exp.get_model().to(device)
    model.eval()
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if fuse:
        model = fuse_model(model)
    if half:
        model = model.half()

    extractor = FeatureExtractor(
        model_name=model_name,
        model_path=model_path,
        device='cuda'
    )
    # extractor = Extractor("/home/jia/PycharmProjects/Yolov5_DeepSort/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")

    for ind, im in enumerate(sorted(os.listdir(imgs_dir))):
        index = ind + 1
        if index < start_frame:
            continue
        if index > end_frame:
            break
        img = cv2.imread(os.path.join(imgs_dir, im))
        height, width = img.shape[:-1]
        # input, ratio = preproc(img, img_size, mean, std)
        # input = torch.from_numpy(input).unsqueeze(0).float().to(device)
        # input = input.half() if half else input
        # outputs = model(input)
        # outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)
        # det = outputs[0].cpu().detach().numpy()
        # det[:, :4] /= ratio
        # rows_to_remove = np.any(det[:, 0:4] < 1, axis=1)  # remove edge detection
        # det = det[~rows_to_remove]
        det = get_det_from_text(text_file, index)
        cropped_imgs = [img[max(0, int(y1)):min(height, int(y2)), max(0, int(x1)):min(width, int(x2))] for
                        x1, y1, x2, y2, _, _, _ in det]
        feature_maps = extractor(cropped_imgs)
        feature_map_np = feature_maps.cpu().numpy()  # (N, C, H', W')
        save_path = os.path.join(output_dir, f"attention_{os.path.splitext(im)[0]}.jpg")
        visualize_feature_maps(img, det, cropped_imgs, feature_map_np, save_path)




