import numpy as np
import os
import torch
import pickle

from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger
from tqdm import tqdm

from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
# jjh
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

from Tracklet import Tracklet

import argparse
# jjh
import torch.nn.functional as F
from TrackletMerger import TrackletMerger


def find_consecutive_segments(track_times):
    """
    Identifies and returns the start and end indices of consecutive segments in a list of times.

    Args:
        track_times (list): A list of frame times (integers) representing when a tracklet was detected.

    Returns:
        list of tuples: Each tuple contains two integers (start_index, end_index) representing the start and end of a consecutive segment.
    """
    segments = []
    start_index = 0
    end_index = 0
    for i in range(1, len(track_times)):
        if track_times[i] == track_times[end_index] + 1:
            end_index = i
        else:
            segments.append((start_index, end_index))
            start_index = i
            end_index = i
    segments.append([start_index, end_index])
    return segments

def split_by_time_seg(tracklets, trklet, tid, time_seg, new_id):
    for ind, seg in enumerate(time_seg):
        if ind == 0:
            tracklets[tid] = trklet.extract(seg[0], seg[1])
        else:
            tmp_frames = trklet.times[seg[0]: seg[1] + 1]
            tmp_scores = trklet.scores[seg[0]: seg[1] + 1]
            tmp_bboxes = trklet.bboxes[seg[0]: seg[1] + 1]
            tmp_embs = trklet.features[seg[0]: seg[1] + 1]
            tmp_feat_scores = trklet.feat_scores[seg[0]: seg[1] + 1]
            tracklets[new_id] = Tracklet(new_id, tmp_frames, tmp_scores,
                                         tmp_bboxes, feats=tmp_embs,
                                         feat_scores=tmp_feat_scores)
            new_id += 1
    return tracklets, new_id

def segment_all_tracks(tmp_trklets, min_len=20):
    new_id = max(tmp_trklets.keys()) + 1
    tracklets = defaultdict()
    for tid in tqdm(sorted(list(tmp_trklets.keys())), total=len(tmp_trklets), desc="segment tracklets"):
        trklet = tmp_trklets[tid]
        if len(trklet.times) < min_len:  # NOTE: Set tracklet length threshold to filter out short ones
            tracklets[tid] = trklet
        else:
            track_times = trklet.times
            time_seg = find_consecutive_segments(track_times)
            tracklets, new_id = split_by_time_seg(tracklets, trklet, tid, time_seg, new_id)
    return tracklets


# jjh
def adjust_segment_boundaries(segments, exit_zero_frame):
    """
    调整轨迹片段边界，将低置信度区间包含到相邻高置信度片段中

    参数:
        segments: 原始高置信度片段列表，格式为[(start1, end1), (start2, end2), ...]

    返回:
        adjusted_segments: 调整后的轨迹段列表
    """
    # 如果没有片段，返回空列表
    if not segments:
        return []

    # 按起始帧排序
    segments = sorted(segments, key=lambda x: x[0])

    adjusted_segments = []

    # 处理第一个片段
    first_start, first_end = segments[0]
    # 如果刚开始就是外观置信度很小的话，要把最开始的帧也加进去
    if first_start != 0 and exit_zero_frame:
        adjusted_segments.append((0, first_start - 1))
    # 第一个片段的结束帧设为下一个片段的起始帧减1
    if len(segments) > 1:
        next_start = segments[1][0]
        adjusted_segments.append((first_start, next_start - 1))
    else:
        adjusted_segments.append(segments[0])

    # 处理中间片段
    for i in range(1, len(segments) - 1):
        current_start, current_end = segments[i]
        next_start = segments[i + 1][0]
        adjusted_segments.append((current_start, next_start - 1))

    # 处理最后一个片段
    if len(segments) > 1:
        last_start, last_end = segments[-1]
        # 最后一个片段的起始帧保持不变
        adjusted_segments.append((last_start, last_end))

    return adjusted_segments


# jjh
def split_trajectory(trklet, low_confidence_thresh=0.7, similarity_thresh=0.7, min_segment_length=5):
    """
    根据外观置信度分离轨迹

    参数:
        trklet: 轨迹对象，包含以下属性:
            features: 外观特征列表 [N, 512]
            times: 帧索引列表 [N]
            bboxes: 边界框列表 [N, 4]
            scores: 检测置信度列表 [N]
            feat_scores: 外观置信度列表 [N]
        low_confidence_thresh: 低置信度阈值
        similarity_thresh: 相似度阈值
        min_segment_length: 最小有效段长度

    返回:
        segments: 分离后的轨迹段列表，每个元素为(start_frame, end_frame)
    """
    # 提取轨迹数据
    embs = np.stack(trklet.features)
    frames = np.array(trklet.times)
    feat_scores = np.array(trklet.feat_scores)

    # 1. 找出所有低置信度点
    low_confidence_indices = np.where(feat_scores < low_confidence_thresh)[0]

    # 如果没有低置信度点，返回整个轨迹
    if len(low_confidence_indices) == 0:
        return [(frames[0], frames[-1])]

    # 2. 根据低置信度点分割轨迹
    segments = []
    start_idx = 0

    for i, idx in enumerate(low_confidence_indices):
        # 跳过过短的段
        if idx - start_idx >= min_segment_length:
            segments.append((start_idx, idx - 1))
        start_idx = idx + 1

    # 添加最后一段
    if len(frames) - start_idx >= min_segment_length:
        segments.append((start_idx, len(frames) - 1))

    # 如果没有有效段，返回空列表
    if len(segments) == 0:
        return False, []

    # 3. 合并相似段
    merged_segments = []
    current_segment = segments[0]

    for i in range(1, len(segments)):
        prev_segment = segments[i - 1]
        next_segment = segments[i]

        # 提取前一段的尾部特征（高置信度部分）
        prev_end = prev_segment[1]
        prev_start = max(prev_segment[0], prev_end - min_segment_length + 1)
        prev_embs = embs[prev_start:prev_end + 1]

        # 提取后一段的头部特征（高置信度部分）
        next_start = next_segment[0]
        next_end = min(next_segment[1], next_start + min_segment_length - 1)
        next_embs = embs[next_start:next_end + 1]

        # 计算平均特征向量的余弦相似度
        prev_avg = np.mean(prev_embs, axis=0).reshape(1, -1)
        next_avg = np.mean(next_embs, axis=0).reshape(1, -1)
        similarity = cosine_similarity(prev_avg, next_avg)[0][0]

        # 如果相似度高，合并段
        if similarity > similarity_thresh:
            current_segment = (current_segment[0], next_segment[1])
        else:
            merged_segments.append(current_segment)
            current_segment = next_segment

    merged_segments.append(current_segment)
    exit_zero_frame = frames[0] == 0
    merged_low_segments = adjust_segment_boundaries(merged_segments, exit_zero_frame)

    return len(merged_low_segments) > 1, merged_low_segments


def split_tracklets(tmp_trklets, len_thres=None):
    """
    Splits each tracklet into multiple tracklets based on an internal distance threshold.

    Args:
        tmp_trklets (dict): Dictionary of tracklets to be processed.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        len_thres (int): Length threshold to filter out short tracklets.
        max_k (int): Maximum number of clusters to consider.

    Returns:
        dict: New dictionary of tracklets after splitting.
    """
    new_id = max(tmp_trklets.keys()) + 1
    tracklets = defaultdict()
    # Splitting algorithm to process every tracklet in a sequence
    for tid in tqdm(sorted(list(tmp_trklets.keys())), total=len(tmp_trklets), desc="Splitting tracklets"):
        # print("Track ID:\n", tid)               # debug line, delete later
        trklet = tmp_trklets[tid]
        if len(trklet.times) < len_thres:  # NOTE: Set tracklet length threshold to filter out short ones
            tracklets[tid] = trklet
        else:
            # jjh 不使用聚类方法进行分离，而是通过高置信度外观进行分离
            id_switch_detected, clusters = split_trajectory(trklet=trklet)
            clusters = list(map(lambda segment: (int(segment[0]), int(segment[1])), clusters))

            # jjh 通过clusters将轨迹进行分离，除了第一个轨迹之外，其他轨迹的id变成新的id
            if not id_switch_detected:
                tracklets[tid] = trklet
            else:
                for cluster in clusters:
                    if cluster[0] == 0:
                        tracklets[tid] = trklet.extract(cluster[0], cluster[1])
                    else:
                        tracklets, new_id = split_by_time_seg(tracklets, trklet, tid, clusters, new_id)

    assert len(tracklets) >= len(tmp_trklets)
    return tracklets


def save_results(sct_output_path, tracklets):
    """
    Saves the final tracklet results into a specified path.

    Args:
        sct_output_path (str): Path where the results will be saved.
        tracklets (dict): Dictionary of tracklets containing their final states.

    """
    results = []

    for i, tid in enumerate(sorted(tracklets.keys())):  # add each track to results
        track = tracklets[tid]
        tid = i + 1
        for instance_idx, frame_id in enumerate(track.times):
            bbox = track.bboxes[instance_idx]

            results.append(
                [frame_id, tid, bbox[0], bbox[1], bbox[2], bbox[3], 1, -1, -1, -1]
            )
    results = sorted(results, key=lambda x: x[0])
    txt_results = []
    for line in results:
        txt_results.append(
            f"{line[0]},{line[1]},{line[2]:.2f},{line[3]:.2f},{line[4]:.2f},{line[5]:.2f},{line[6]},{line[7]},{line[8]},{line[9]}\n"
        )

    # NOTE: uncomment to save results
    with open(sct_output_path, 'w') as f:
        f.writelines(txt_results)
    logger.info(f"save SCT results to {sct_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Global tracklet association with splitting and connecting.")
    parser.add_argument('--dataset',
                        type=str,
                        default="SoccerNet",
                        help='Dataset name (e.g., SportsMOT, SoccerNet).')

    parser.add_argument('--tracker',
                        type=str,
                        default="DeepEIoU",
                        help='Tracker name.')

    parser.add_argument('--track_src',
                        type=str,
                        default="/home/jia/PycharmProjects/gta-link/test_data/DeepEIoU_Tracklets_test_data",
                        help='Source directory of tracklet pkl files.'
                        )

    parser.add_argument('--use_split',
                        action='store_false',
                        help='If using split component.')

    parser.add_argument('--min_len',
                        type=int,
                        default=20,
                        help='Minimum length for a tracklet required for splitting.')

    parser.add_argument('--use_connect',
                        action='store_false',
                        help='If using connecting component.')

    parser.add_argument('--merge_dist_thres',
                        type=float,
                        default=0.4,
                        help='Minimum cosine distance between two tracklets for merging.')
    return parser.parse_args()


def main():
    args = parse_args()
    # Determine the process based on the flags
    if args.use_split and args.use_connect:
        process = "Split+Connect"
    elif args.use_split:
        process = "Split"
    elif args.use_connect:
        process = "Connect"
    else:
        raise ValueError("Both use_split and use_connect are false, must at least use connect.")

    seq_tracks_dir = args.track_src
    data_path = os.path.dirname(seq_tracks_dir)
    seqs_tracks = os.listdir(seq_tracks_dir)

    tracker = args.tracker
    dataset = args.dataset

    seqs_tracks.sort()

    process_limit = 10000  # debug line, delete later
    for seq_idx, seq in enumerate(seqs_tracks):
        if seq_idx >= process_limit:  # debug line, delete later
            break  # debug line, delete later

        seq_name = seq.split('.')[0]
        logger.info(f"Processing seq {seq_idx + 1} / {len(seqs_tracks)}")
        with open(os.path.join(seq_tracks_dir, seq), 'rb') as pkl_f:
            tmp_trklets = pickle.load(pkl_f)  # dict(key:track id, value:tracklet)

        # jjh 断开所有不连续的track
        tmp_trklets = segment_all_tracks(tmp_trklets, min_len=args.min_len)


        if args.use_split:
            print(f"----------------Number of tracklets before splitting: {len(tmp_trklets)}----------------")
            splitTracklets = split_tracklets(tmp_trklets, len_thres=args.min_len)
        else:
            splitTracklets = tmp_trklets

        tracklets_merger = TrackletMerger()
        merged_list = tracklets_merger.process_all_tracks(splitTracklets)
        mergedTracklets = defaultdict()
        for track in merged_list:
            mergedTracklets[track.track_id] = track

        sct_name = f'{tracker}_{dataset}_{process}'
        os.makedirs(os.path.join(data_path, sct_name), exist_ok=True)
        new_sct_output_path = os.path.join(data_path, sct_name, '{}.txt'.format(seq_name))
        save_results(new_sct_output_path, mergedTracklets)


if __name__ == "__main__":
    main()