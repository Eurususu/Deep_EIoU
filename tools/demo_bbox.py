import argparse
import os
import os.path as osp
import numpy as np
import time
import cv2
import torch
import sys
from utils import check_path_type, get_sorted_image_files, parse_detection_file
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer

from tracker.Deep_EIoU import Deep_EIoU
from reid.torchreid.utils import FeatureExtractor
import torchvision.transforms as T

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("DeepEIoU Demo")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default=f"{PROJECT_ROOT}/videos/football2.mp4", # /home/jia/PycharmProjects/gta-link/SoccerNet/tracking-2023/test/SNMOT-116
        help="path to images or video"
    )
    parser.add_argument(
        "--save_result",
        default=True,
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=f"{PROJECT_ROOT}/yolox/yolox_x_ch_sportsmot.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float,
                        help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--nms_thres", type=float, default=0.7, help='nms threshold')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # reid args
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')
    return parser


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def imageflow_demo(dets_txt, extractor, vis_folder, current_time, args):
    data_type = check_path_type(args.path)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, args.path.split("/")[-1])
    logger.info(f"video or images save_path is {save_path}")
    tracker = Deep_EIoU(args, frame_rate=30)
    dets = parse_detection_file(dets_txt)
    if data_type == "video":
        cap = cv2.VideoCapture(args.path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        timer = Timer()
        frame_id = 0
        results = []
        while True:
            if frame_id % 30 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            ret_val, frame = cap.read()
            if ret_val:
                det = np.array(dets[frame_id + 1])
                if len(det) > 0:
                    det[:, 2:4] = det[:, 0:2] + det[:, 2:4]
                    rows_to_remove = np.any(det[:, 0:4] < 1, axis=1)  # remove edge detection
                    det = det[~rows_to_remove]
                    cropped_imgs = [frame[max(0, int(y1)):min(height, int(y2)), max(0, int(x1)):min(width, int(x2))] for
                                    x1, y1, x2, y2, _, in det]
                    embs = extractor(cropped_imgs)
                    embs = embs.cpu().detach().numpy()
                    online_targets = tracker.update(det, embs)
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.last_tlwh
                        tid = t.track_id
                        if tlwh[2] * tlwh[3] > args.min_box_area:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            results.append(
                                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                    timer.toc()
                    online_im = plot_tracking(
                        frame.copy(), online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                    )
                else:
                    timer.toc()
                    online_im = frame.copy()
                if args.save_result:
                    vid_writer.write(online_im)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break
            frame_id += 1

        if args.save_result:
            res_file = osp.join(vis_folder, f"{timestamp}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")
    elif data_type == "directory":
        img_list = get_sorted_image_files(args.path, recursive=True)
        timer = Timer()
        results = []
        for frame_id, img in enumerate(img_list):
            frame = cv2.imread(img)
            height, width = frame.shape[:-1]
            if frame_id % 30 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            det = np.array(dets[frame_id])
            if len(det) > 0:
                det[:, 2:4] = det[:, 0:2] + det[:, 2:4]
                rows_to_remove = np.any(det[:, 0:4] < 1, axis=1)  # remove edge detection
                det = det[~rows_to_remove]
                cropped_imgs = [frame[max(0, int(y1)):min(height, int(y2)), max(0, int(x1)):min(width, int(x2))] for
                                x1, y1, x2, y2, _, _, _ in det]
                embs = extractor(cropped_imgs)
                embs = embs.cpu().detach().numpy()
                online_targets = tracker.update(det, embs)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.last_tlwh
                    tid = t.track_id
                    if tlwh[2] * tlwh[3] > args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    frame.copy(), online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = frame.copy()
            # if args.save_result:
            #     img_name = os.path.basename(img)
            #     os.makedirs(save_path, exist_ok=True)
            #     save_name = os.path.join(save_path, img_name)
            #     cv2.imwrite(save_name, online_im)
        if args.save_result:
            res_file = osp.join(vis_folder, f"{os.path.basename(save_path)}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_vis")
    os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # model = exp.get_model().to(args.device)
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    # model.eval()

    # if not args.trt:
    #     if args.ckpt is None:
    #         ckpt_file = "checkpoints/best_ckpt.pth.tar"
    #     else:
    #         ckpt_file = args.ckpt
    #     logger.info("loading checkpoint")
    #     ckpt = torch.load(ckpt_file, map_location="cpu")
    #     # load the model state dict
    #     model.load_state_dict(ckpt["model"])
    #     logger.info("loaded checkpoint done.")
    #
    # if args.fuse:
    #     logger.info("\tFusing model...")
    #     model = fuse_model(model)
    #
    # if args.fp16:
    #     model = model.half()  # to FP16
    #
    # if args.trt:
    #     assert not args.fuse, "TensorRT model is not support model fusing!"
    #     trt_file = osp.join(output_dir, "model_trt.pth")
    #     assert osp.exists(
    #         trt_file
    #     ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
    #     model.head.decode_in_inference = False
    #     decoder = model.head.decode_outputs
    #     logger.info("Using TensorRT to inference")
    # else:
    #     trt_file = None
    #     decoder = None

    current_time = time.localtime()

    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=f'{PROJECT_ROOT}/checkpoints/sports_model.pth.tar-60',
        device='cuda'
    )
    dets_txt = "/home/jia/PycharmProjects/Deep-EIoU/football2_filter.txt"
    imageflow_demo(dets_txt, extractor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
