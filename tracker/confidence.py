import numpy as np


def batch_giou(boxes):
    """
    批量计算所有检测框之间的GIoU矩阵
    :param boxes: 所有检测框坐标 [N, 4] (x1, y1, x2, y2)
    :return: GIoU矩阵 [N, N]
    """
    N = boxes.shape[0]

    # 扩展维度用于广播计算 [N,1,4] 和 [1,N,4]
    boxes1 = boxes[:, None, :]  # [N,1,4]
    boxes2 = boxes[None, :, :]  # [1,N,4]

    # 计算交集区域
    inter_x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

    # 计算交集面积
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # 计算各自面积
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算并集面积
    union_area = area1 + area2 - inter_area

    # 计算最小闭包区域 (Convex Hull)
    ch_x1 = np.minimum(boxes1[..., 0], boxes2[..., 0])
    ch_y1 = np.minimum(boxes1[..., 1], boxes2[..., 1])
    ch_x2 = np.maximum(boxes1[..., 2], boxes2[..., 2])
    ch_y2 = np.maximum(boxes1[..., 3], boxes2[..., 3])

    # 计算闭包区域面积
    ch_area = (ch_x2 - ch_x1) * (ch_y2 - ch_y1)

    # 计算IoU (避免除零)
    iou = np.divide(inter_area, union_area, where=union_area > 0, out=np.zeros_like(inter_area))

    # 计算GIoU = IoU - (C - U)/C
    giou = iou - np.divide(ch_area - union_area, ch_area, where=ch_area > 0, out=np.zeros_like(iou))

    # 确保GIoU范围在[-1,1]之间
    return np.clip(giou, -1.0, 1.0)


class EfficientAdaptiveAppearanceUpdater:
    def __init__(self, base_update_rate=0.3):
        """
        初始化外观更新器
        :param base_update_rate: 基础外观更新率 (0~1)
        """
        self.base_update_rate = base_update_rate

    def compute_appearance_confidence(self, boxes, confidences, ignore_th=0.2):
        """
        批量计算所有检测框的外观置信度
        :param boxes: 所有检测框坐标 [N, 4] (x1, y1, x2, y2)
        :param confidences: 检测框置信度 [N]
        :return: 每个检测框的外观置信度 [N]
        """
        # 计算GIoU矩阵 [N, N]
        giou_matrix = batch_giou(boxes)

        # 计算每个框与其他框的总GIoU (排除自身)
        np.fill_diagonal(giou_matrix, 0)  # 将对角线设为0
        giou_matrix[giou_matrix < ignore_th] = 0
        total_giou = np.sum(giou_matrix, axis=1)

        non_zero_counts = np.sum(giou_matrix > 0, axis=1)

        effective_means = np.zeros((giou_matrix.shape[0],))
        with np.errstate(divide='ignore', invalid='ignore'):
            effective_means = np.where(non_zero_counts > 0,
                                       total_giou / non_zero_counts,
                                       0)

        return confidences*(1 - effective_means)

    def update_features(self, current_features, track_features,
                        boxes, confidences):
        """
        批量更新所有轨迹的外观特征
        :param current_features: 当前帧的外观特征 [N, D]
        :param track_features: 轨迹的外观特征 [N, D]
        :param boxes: 检测框坐标 [N, 4]
        :param confidences: 检测框置信度 [N]
        :return: 更新后的外观特征 [N, D], 实际更新率 [N], 外观置信度 [N]
        """
        # 计算外观置信度
        app_confidences = self.compute_appearance_confidence(boxes, confidences)

        # 计算实际更新率
        effective_rates = self.base_update_rate * app_confidences

        # 使用指数移动平均更新外观特征
        updated_features = (
                (1 - effective_rates[:, None]) * track_features +
                effective_rates[:, None] * current_features
        )

        return updated_features, effective_rates, app_confidences