import os
import json
import torch
import random
import cv2
import math
import numpy as np
import torchvision
from pathlib import Path
from torch import nn
from torch import optim
import torch.utils.data as Data
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt 
import matplotlib
import tqdm
from sklearn.metrics import jaccard_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    res = (intersection + 1e-15) / (union + 1e-15)
    return res.detach().cpu().numpy()



def get_Dice(y_true, y_pred):
    res = (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)
    return res.detach().cpu().numpy()



def testing_score(model, testloader):
    tot_Jaccard = 0.0
    tot_dice = 0.0
    model.eval()
    
    for data, mask, temp, path in testloader:
        data, mask, temp, path = data.to(device), mask.float().to(device), temp.to(device), path[0]
        output = model(data,temp, path)
        target = mask.transpose(0,1)
        preds = torch.sigmoid(output) > 0.5
        preds = preds.to(torch.float32)
        preds = preds.to(torch.float32)
        tot_Jaccard += get_jaccard(target[0][0], preds[0][0])
        tot_dice += get_Dice(target[0][0], preds[0][0])

    mean_Jaccard = tot_Jaccard / len(testloader.dataset)
    mean_dice = tot_dice / len(testloader.dataset)
    return mean_Jaccard, mean_dice