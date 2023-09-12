import copy
from pathlib import Path
import time
from typing import List
from box import Box
import torch.nn.functional as F
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
from tqdm import tqdm
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


def replace_file_line(filename: str | Path, old_string: str, new_string: str):
    "Adopted from https://stackoverflow.com/questions/4128144/replace-string-within-file-contents"
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)
        
        
def test_inference_speed(input_model, device: str | torch.device = "cpu", input_size: int = 224, iterations: int = 1000):
    # cuDnn configurations
    actual_cuddn_benchmark = torch.backends.cudnn.benchmark
    actual_cudnn_deterministic = torch.backends.cudnn.deterministic
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    model = copy.deepcopy(input_model).to(device)

    model.eval()

    time_list = []
    for i in tqdm(range(iterations+1), desc="Testing inference time"):
        random_input = torch.randn(1,3,input_size,input_size).to(device)
        torch.cuda.synchronize()
        tic = time.perf_counter()
        model(random_input)
        torch.cuda.synchronize()
        # the first iteration time cost much higher, so exclude the first iteration
        #print(time.time()-tic)
        time_list.append(time.perf_counter()-tic)
    time_list = time_list[1:]
    
    torch.backends.cudnn.benchmark = actual_cuddn_benchmark
    torch.backends.cudnn.deterministic = actual_cudnn_deterministic
    
    return sum(time_list)/10000