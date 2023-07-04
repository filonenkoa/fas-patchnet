from pathlib import Path
from typing import List
from box import Box
import torch.nn.functional as F
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
import yaml
from torch import optim
import torch.distributed as dist
import cv2


def calc_acc(pred, target):
    pred = torch.argmax(pred, dim=1)
    equal = torch.mean(pred.eq(target).type(torch.FloatTensor))
    return equal.item()


def compute_eer(labels, scores):
    """Compute the Equal Error Rate (EER) from the predictions and scores.
    Args:
        labels (list[int]): values indicating whether the ground truth
            value is positive (1) or negative (0).
        scores (list[float]): the confidence of the prediction that the
            given sample is a positive.
    Return:
        (float, thresh): the Equal Error Rate and the corresponding threshold
    NOTES:
       The EER corresponds to the point on the ROC curve that intersects
       the line given by the equation 1 = FPR + TPR.
       The implementation of the function was taken from here:
       https://yangcha.github.io/EER-ROC/
    """
    scores_lst = [val[labels[idx]].int() for idx, val in enumerate(scores)]
    fpr, tpr, thresholds = roc_curve(labels.tolist(), scores_lst, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    labels = labels.to("cuda")
    scores = scores.to("cuda")
    return eer, thresh


def read_cfg(cfg_file: str) -> Box:
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (Box): configuration in Box dict wrapper
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return Box(cfg)


def get_device(cfg):
    device = None
    if cfg['device'] == 'cpu':
        device = torch.device("cpu")
    elif cfg['device'].startswith("cuda"):
        device = torch.device(cfg['device'])
    else:
        raise NotImplementedError
    return device


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def frame_count(video_path, manual=False):
    def manual_count(handler):
        frames = 0
        while True:
            status, frame = handler.read()
            if not status:
                break
            frames += 1
        return frames 

    cap = cv2.VideoCapture(video_path)
    # Slow, inefficient but 100% accurate method 
    if manual:
        frames = manual_count(cap)
    # Fast, efficient but inaccurate method
    else:
        try:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            frames = manual_count(cap)
    cap.release()
    return frames


def get_all_file_paths(path: Path, extensions=[".jpg", ".png", ".jpeg", ".bmp"]) -> List[Path]:
    path = path.expanduser()
    files_paths: List[Path] = []
    for extension in extensions:
        files = list(path.rglob(f"*{extension}"))
        files_paths.extend(files)
    return files_paths