import os
import json
import torch
import cv2
import math
import numpy as np
from pathlib import Path
import torch.utils.data as Data
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F




def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    res = (intersection + 1e-15) / (union + 1e-15)
    return res.detach().cpu().numpy()


def dice(y_true, y_pred):
    res = (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)
    return res.detach().cpu().numpy()

def get_Dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0
            
    classes = y_true.unique().detach().cpu().tolist()
    for instrument_id in classes:
        if instrument_id == 0:
            continue
        result.append(dice(y_true == instrument_id, y_pred == instrument_id))

    return np.mean(result)

def get_Jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    classes = y_true.unique().detach().cpu().tolist()
    for instrument_id in classes:
        if instrument_id == 0:
            continue
        result.append(jaccard(y_true == instrument_id, y_pred == instrument_id))

    return np.mean(result)