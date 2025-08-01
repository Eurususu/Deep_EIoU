import numpy as np
import motmetrics as mm
import os
import os.path as osp
from typing import Union
try:
    import seaborn as sns
except ImportError:
    sns = None
import cv2
from tqdm import tqdm

def imshow_mot_errors(*args, backend: str = 'cv2', **kwargs):
    """Show the wrong tracks on the input image.

    Args:
        backend (str, optional): Backend of visualization.
            Defaults to 'cv2'.
    """
    if backend == 'cv2':
        return _cv2_show_wrong_tracks(*args, **kwargs)
    else:
        raise NotImplementedError()


def _cv2_show_wrong_tracks(img: Union[str, np.ndarray],
                           bboxes: np.ndarray,
                           ids: np.ndarray,
                           error_types: np.ndarray,
                           thickness: int = 2,
                           font_scale: float = 0.4,
                           text_width: int = 10,
                           text_height: int = 15,
                           show: bool = False,
                           wait_time: int = 10,
                           out_file: str = None) -> np.ndarray:
    """Show the wrong tracks with opencv.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): A ndarray of shape (k, 5).
        ids (ndarray): A ndarray of shape (k, ).
        error_types (ndarray): A ndarray of shape (k, ), where 0 denotes
            false positives, 1 denotes false negative and 2 denotes ID switch.
        thickness (int, optional): Thickness of lines.
            Defaults to 2.
        font_scale (float, optional): Font scale to draw id and score.
            Defaults to 0.4.
        text_width (int, optional): Width to draw id and score.
            Defaults to 10.
        text_height (int, optional): Height to draw id and score.
            Defaults to 15.
        show (bool, optional): Whether to show the image on the fly.
            Defaults to False.
        wait_time (int, optional): Value of waitKey param.
            Defaults to 100.
        out_file (str, optional): The filename to write the image.
            Defaults to None.

    Returns:
        ndarray: Visualized image.
    """
    if sns is None:
        raise ImportError('please run pip install seaborn')
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert ids.ndim == 1 or ids.ndim == 2, \
        f' ids ndim should be 1, but its ndim is {ids.ndim}.'
    assert error_types.ndim == 1, \
        f' error_types ndim should be 1, but its ndim is {error_types.ndim}.'
    assert bboxes.shape[0] == ids.shape[0], \
        'bboxes.shape[0] and ids.shape[0] should have the same length.'
    assert bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 5, but its {bboxes.shape[1]}.'

    bbox_colors = sns.color_palette()
    # red, yellow, blue
    bbox_colors = [bbox_colors[3], bbox_colors[1], bbox_colors[0]]
    bbox_colors = [[int(255 * _c) for _c in bbox_color][::-1]
                   for bbox_color in bbox_colors]

    if isinstance(img, str):
        img = cv2.imread(img)
    else:
        assert img.ndim == 3

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    for bbox, error_type, id in zip(bboxes, error_types, ids):
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = float(bbox[-1])

        # bbox
        bbox_color = bbox_colors[error_type]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        # FN does not have id and score
        # if error_type == 1:
        #     continue

        # score
        text = '{:.02f}'.format(score)
        width = (len(text) - 1) * text_width
        img[y1:y1 + text_height, x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            text, (x1, y1 + text_height - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

        # id
        if isinstance(id, np.int32):
            text = str(id)
        else:
            assert len(id) == 2
            text = f'{id[0]}->{id[1]}'
        width = len(text) * text_width
        img[y1 + text_height:y1 + text_height * 2,
            x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            text, (x1, y1 + text_height * 2 - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

    if show:
        cv2.imshow('image', img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, img)
    return img


gt_file = 'gt.txt'
res_file = 'pred.txt'
img_dir = '/home/jia/PycharmProjects/gta-link/test_data/1212/img1'
out_dir = '/home/jia/PycharmProjects/Deep-EIoU/vis_result/img1'
log = 'log.txt'
gt = mm.io.loadtxt(gt_file)
res = mm.io.loadtxt(res_file)
acc = mm.utils.compare_to_groundtruth(gt, res)
filenames_dict = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if i.endswith(".jpg")]
filenames_dict = sorted(filenames_dict)

frames_id_list = sorted(
    list(set(acc.mot_events.index.get_level_values(0))))

for frame_id in tqdm(frames_id_list):
    # events in the current frame
    events = acc.mot_events.xs(frame_id)
    cur_res = res.loc[frame_id] if frame_id in res.index else None
    cur_gt = gt.loc[frame_id] if frame_id in gt.index else None
    # path of image
    img = filenames_dict[frame_id]
    fps = events[events.Type == 'FP']
    fns = events[events.Type == 'MISS']
    idsws = events[events.Type == 'SWITCH']

    bboxes, ids, error_types = [], [], []
    for fp_index in fps.index:
        hid = events.loc[fp_index].HId
        bboxes.append([
            cur_res.loc[hid].X, cur_res.loc[hid].Y,
            cur_res.loc[hid].X + cur_res.loc[hid].Width,
            cur_res.loc[hid].Y + cur_res.loc[hid].Height,
            cur_res.loc[hid].Confidence
        ])
        ids.append(hid)
        # error_type = 0 denotes false positive error
        error_types.append(0)
        with open (log, 'a') as f:
            text = f'在第{frame_id}帧发生误检，误检的id为{hid}\n'
            f.write(text)
    for fn_index in fns.index:
        oid = events.loc[fn_index].OId
        bboxes.append([
            cur_gt.loc[oid].X, cur_gt.loc[oid].Y,
            cur_gt.loc[oid].X + cur_gt.loc[oid].Width,
            cur_gt.loc[oid].Y + cur_gt.loc[oid].Height,
            cur_gt.loc[oid].Confidence
        ])
        ids.append(oid)
        # error_type = 1 denotes false negative error
        error_types.append(1)
        with open (log, 'a') as f:
            text = f'在第{frame_id}帧发生漏检，漏检的id为{oid}\n'
            f.write(text)
    for idsw_index in idsws.index:
        hid = events.loc[idsw_index].HId
        oid = events.loc[idsw_index].OId
        bboxes.append([
            cur_res.loc[hid].X, cur_res.loc[hid].Y,
            cur_res.loc[hid].X + cur_res.loc[hid].Width,
            cur_res.loc[hid].Y + cur_res.loc[hid].Height,
            cur_res.loc[hid].Confidence
        ])
        ids.append([oid, hid])
        # error_type = 2 denotes id switch
        error_types.append(2)
        with open (log, 'a') as f:
            text = f'在第{frame_id}帧发生id switch，从{oid}变成了{hid}\n'
            f.write(text)
    if len(bboxes) == 0:
        bboxes = np.zeros((0, 5), dtype=np.float32)
    else:
        bboxes = np.asarray(bboxes, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int32)
    error_types = np.asarray(error_types, dtype=np.int32)
    # imshow_mot_errors(
    #     img,
    #     bboxes,
    #     ids,
    #     error_types,
    #     show=False,
    #     out_file=None,  # osp.join(out_dir, f'{frame_id:06d}.jpg'),
    #     backend="cv2")  # cv2或plt

