import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict


class TrackletMerger:
    def __init__(self,
                 max_time_gap=30,  # 最大允许时间间隔(帧)
                 min_confidence=0.7,  # 外观高置信度阈值
                 ):

        self.max_time_gap = max_time_gap
        self.min_confidence = min_confidence
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 默认配置参数
        self.config = {
            'spatial_threshold_small_gap': 50,  # 小时间间隔下的空间距离阈值
            'motion_threshold_small_gap': 30,  # 小时间间隔下的运动距离阈值
            'appearance_threshold_small_gap': 0.25,  # 小时间间隔下的外观距离阈值
            'appearance_threshold_large_gap': 0.35,  # 大时间间隔下的外观距离阈值
            'confidence_threshold': 0.6,  # 轨迹置信度阈值
            'min_match_quality': 0.7,  # 最小匹配质量
            'max_candidates': 5,  # 考虑的最佳候选数量
        }

        # 轨迹段存储
        self.track_segments = []

    def _compute_velocity(self, track, last_n_frames=5):
        """计算轨迹最后几帧的平均速度向量"""
        boxes = track.bboxes
        n_frames = len(boxes)

        # 如果轨迹帧数不足，使用所有可用帧
        if n_frames <= last_n_frames:
            start_index = 0
        else:
            start_index = n_frames - last_n_frames

        # 计算中心点
        centers = []
        for i in range(start_index, n_frames):
            box = boxes[i]
            x1, y1, w, h = box
            centers.append([x1 + w / 2, y1 + h / 2])

        # 计算最近几帧的平均速度
        total_delta = [0, 0]
        frame_count = 0

        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            total_delta[0] += dx
            total_delta[1] += dy
            frame_count += 1

        if frame_count == 0:
            return [0, 0]  # 避免除以零

        return [
            total_delta[0] / frame_count,
            total_delta[1] / frame_count
        ]

    def _filter_features(self, track):
        """根据置信度筛选有效特征"""
        valid_features = []
        for feat, conf in zip(track.features, track.feat_scores):
            if conf >= self.min_confidence:
                valid_features.append(feat)

        if not valid_features:
            # 没有有效特征时使用所有特征均值
            return np.mean(track.features, axis=0)
        return np.mean(valid_features, axis=0)

    def compute_pairwise_distance(self, track1, track2):
        """计算两条轨迹之间的综合距离"""
        # 时间连续性检查
        end1 = max(track1.times)
        start2 = min(track2.times)

        # 时间不连续时直接返回大距离
        if end1 >= start2:
            spatial_dist = float('inf')
            motion_dist = float("inf")
            appearance_dist = float("inf")
            combined_dist = {"spatial_dist": spatial_dist, "motion_dist": motion_dist,
                             "appearance_dist": appearance_dist, "time_dist": -1}
            return combined_dist

        # 计算时间间隔
        time_gap = start2 - end1
        if time_gap < self.max_time_gap:
            # 1. 空间连续性计算
            last_box1 = track1.bboxes[-1]
            first_box2 = track2.bboxes[0]

            # 计算中心点距离
            center1 = [last_box1[0] + last_box1[2] / 2, last_box1[1] + last_box1[3] / 2]
            center2 = [first_box2[0] + first_box2[2] / 2, first_box2[1] + first_box2[3] / 2]
            spatial_dist = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

            # 2. 运动一致性检查
            velocity1 = self._compute_velocity(track1)
            # 预测位置 = 最后位置 + 速度 * 时间间隔
            predicted_center = [
                center1[0] + velocity1[0] * time_gap,
                center1[1] + velocity1[1] * time_gap
            ]
            motion_dist = np.sqrt((predicted_center[0] - center2[0]) ** 2 + (predicted_center[1] - center2[1]) ** 2)
        else:
            spatial_dist = float('inf')
            motion_dist = float('inf')

        # 3. 外观相似度计算
        feat1 = self._filter_features(track1)
        feat2 = self._filter_features(track2)
        t1 = torch.tensor(feat1, dtype=torch.float32).to(self.device)
        t2 = torch.tensor(feat2, dtype=torch.float32).to(self.device)
        cos_sim = F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0))
        appearance_dist = 1 - cos_sim.item()

        # 综合距离计算 (值越小表示越可能属于同一目标)
        combined_dist = {"spatial_dist": spatial_dist,  "motion_dist": motion_dist,  "appearance_dist": appearance_dist, "time_dist": time_gap}

        return combined_dist

    def _calculate_match_quality(self, dist_dict, segment, track):
        """计算匹配质量分数 (0-1范围)"""
        time_gap = dist_dict['time_dist']

        # 计算轨迹置信度
        seg_conf = np.mean(segment.confidences) if hasattr(segment, 'confidences') else 1.0
        track_conf = np.mean(track.confidences) if hasattr(track, 'confidences') else 1.0
        min_confidence = min(seg_conf, track_conf)

        # 时间间隔因子 (间隔越小，因子越大)
        time_factor = 1.0 - min(time_gap / self.max_time_gap, 1.0)

        # 空间距离因子
        spatial_factor = 1.0 - min(
            dist_dict['spatial_dist'] / self.config['spatial_threshold_small_gap'],
            1.0
        )

        # 运动距离因子
        motion_factor = 1.0 - min(
            dist_dict['motion_dist'] / self.config['motion_threshold_small_gap'],
            1.0
        )

        # 外观距离因子 (考虑置信度)
        if min_confidence > self.config['confidence_threshold']:
            appearance_factor = 1.0 - min(
                dist_dict['appearance_dist'] / self.config['appearance_threshold_small_gap'],
                1.0
            )
        else:
            appearance_factor = 1.0 - min(
                dist_dict['appearance_dist'] / (self.config['appearance_threshold_small_gap'] * 0.5),
                1.0
            )

        # 根据时间间隔调整权重
        if time_gap < 5:  # 小时间间隔
            weights = [0.4, 0.4, 0.2]  # 空间、运动、外观
        elif time_gap < 15:  # 中等时间间隔
            weights = [0.3, 0.3, 0.4]
        elif time_gap <= self.max_time_gap:
            weights = [0.1, 0.1, 0.8]
        else:  # 大时间间隔
            weights = [0.0, 0.0, 1.0]

        # 计算综合质量
        quality = (weights[0] * spatial_factor +
                   weights[1] * motion_factor) * time_factor + weights[2] * appearance_factor

        return quality

    def _is_valid_match(self, dist_dict, segment, track):
        """判断是否有效的匹配"""
        time_gap = dist_dict['time_dist']

        # 计算轨迹置信度
        seg_conf = np.mean(segment.confidences) if hasattr(segment, 'confidences') else 1.0
        track_conf = np.mean(track.confidences) if hasattr(track, 'confidences') else 1.0
        min_confidence = min(seg_conf, track_conf)

        # 小时间间隔匹配标准
        if time_gap < 5:
            spatial_ok = dist_dict['spatial_dist'] < self.config['spatial_threshold_small_gap']
            motion_ok = dist_dict['motion_dist'] < self.config['motion_threshold_small_gap']

            if min_confidence > self.config['confidence_threshold']:
                appearance_ok = dist_dict['appearance_dist'] < self.config['appearance_threshold_small_gap']
            else:
                # 低置信度要求更严格的外观匹配
                appearance_ok = dist_dict['appearance_dist'] < (self.config['appearance_threshold_small_gap'] * 0.7)

            return (spatial_ok or motion_ok) and appearance_ok

        # 中等时间间隔匹配标准
        elif time_gap < 15:
            spatial_ok = dist_dict['spatial_dist'] < (self.config['spatial_threshold_small_gap'] * 1.5)
            motion_ok = dist_dict['motion_dist'] < (self.config['motion_threshold_small_gap'] * 1.5)

            if min_confidence > self.config['confidence_threshold']:
                appearance_ok = dist_dict['appearance_dist'] < (self.config['appearance_threshold_small_gap'] * 0.8)
            else:
                appearance_ok = dist_dict['appearance_dist'] < (self.config['appearance_threshold_small_gap'] * 0.6)

            # 至少满足两个条件，其中外观必须满足
            return appearance_ok and (spatial_ok or motion_ok)

        # 大时间间隔匹配标准
        else:
            # 主要依赖外观距离
            if min_confidence > self.config['confidence_threshold']:
                appearance_ok = dist_dict['appearance_dist'] < self.config['appearance_threshold_large_gap']
            else:
                # 低置信度要求更严格的外观匹配
                appearance_ok = dist_dict['appearance_dist'] < (self.config['appearance_threshold_large_gap'] * 0.8)


            return appearance_ok

    def _merge_tracks(self, segment, new_track):
        """将新轨迹合并到现有段中"""
        # 1. 合并边界框
        segment.bboxes.extend(new_track.bboxes)

        # 2. 合并时间戳
        segment.times.extend(new_track.times)

        # 3. 合并特征
        segment.features.extend(new_track.features)

        # 4. 合并置信度
        segment.scores.extend(new_track.scores)
        segment.feat_scores.extend(new_track.feat_scores)

        # 5. 按时间排序
        sorted_indices = np.argsort(segment.times)
        segment.times = [segment.times[i] for i in sorted_indices]
        segment.bboxes = [segment.bboxes[i] for i in sorted_indices]
        segment.features = [segment.features[i] for i in sorted_indices]
        segment.scores = [segment.scores[i] for i in sorted_indices]
        segment.feat_scores = [segment.feat_scores[i] for i in sorted_indices]

    def add_track(self, track):
        """添加新轨迹到系统"""
        # 如果是第一段轨迹
        if not self.track_segments:
            self.track_segments.append(track)
            return

        # 找到当前轨迹的起始时间
        start_time = min(track.times)


        # 查找可能的匹配段
        candidates = []
        for segment in self.track_segments:
            # 只考虑结束时间早于当前开始时间的段
            if max(segment.times) < start_time:
                dist_dict = self.compute_pairwise_distance(segment, track)

                # 计算匹配质量
                match_quality = self._calculate_match_quality(dist_dict, segment, track)
                if match_quality > 0:
                    candidates.append((segment, dist_dict, match_quality))

        # 如果没有候选，创建新轨迹段
        if not candidates:
            # self.track_segments.append(track)
            if len(track.times) > 30:
                print(len(track.times))
            return

        # 按匹配质量排序候选
        candidates.sort(key=lambda x: x[2], reverse=True)

        # 选择最佳匹配
        # best_match = None
        # for segment, dist_dict, quality in candidates[:self.config['max_candidates']]:
        #     if self._is_valid_match(dist_dict, segment, track):
        #         best_match = segment
        #         break
        best_match = candidates[0][0]

        # 如果找到匹配，合并轨迹
        if best_match:
            self._merge_tracks(best_match, track)
        else:
            self.track_segments.append(track)


    def process_all_tracks(self, tracks):
        """处理所有轨迹"""
        # 按起始时间排序
        tracks_list = list(tracks.values())
        tracks_list.sort(key=lambda t: min(t.times))

        # 找到起始帧为1的轨迹作为初始段
        initial_tracks = [t for t in tracks_list if min(t.times) == 0]
        remaining_tracks = [t for t in tracks_list if min(t.times) > 0]

        # 添加初始段
        for track in initial_tracks:
            self.track_segments.append(track)

        # 处理剩余轨迹
        for track in remaining_tracks:
            self.add_track(track)

        return self.track_segments


