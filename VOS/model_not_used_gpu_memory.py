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
from torch.nn import functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt 
import matplotlib
import tqdm
from PIL import Image

from module import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyTransformer(nn.Module):
    def __init__(self, num_layers=4):
        super(MyTransformer, self).__init__()

        self.backbone = CNNBackbone()
        self.pos_encoding = PositionalEncoding3D()
        self.self_attn1 = SSTEncoder()
        self.self_attn2 = SSTEncoder()
        self.self_attn3 = SSTEncoder()
        self.self_attn4 = SSTEncoder()
        # self.self_attn5 = SSTEncoder()
        # input dim = CNN_out * 2 + num_layers * N_classes
        self.dec1 = DecoderBlock(512*2+4*1,1024,512)
        self.dec2 = DecoderBlock(512,256,256)
        self.dec3 = DecoderBlock(256,128,1)
        # self.conv1 = Upscaler1x1Conv(512*2+5*5,512,2)
        # self.conv2 = Upscaler1x1Conv(512*2+5*5,256,4)
        # self.dec4 = DecoderBlock(128,64,8)
        self.sigmoid = nn.Sigmoid()
        self.cnn_feat = torch.zeros(1)
        self.encod_feat = torch.zeros(1)
        self.attn_scores = torch.zeros(1)
        self.object_affinity = torch.zeros(1)
        self.tau = 4
    

    def forward(self, x, m, c):
        """
        x - input frames sequence;
        m - previous masks prediction;
        c - num of objects
        """
        num_cls = int(c) + 1                              # include background
        _, T, _, H, W = x.shape
        # 1. subtract feature embedding from CNN backbone [b,T,C,H',W']
        cnn_fs = self.backbone(x)
        self.cnn_feat = cnn_fs
        # Get positional encoding and add it to the feature embedding
        pe = self.pos_encoding(cnn_fs)
        encoder_in = pe + cnn_fs
 
        # 2. Encoded feature from transformer layers 
        outputs = []
        for idx in range(1,T):
            if idx < self.tau:
                emb_tau = encoder_in[:,0:idx+1,:,:,:]                   # Slice the embed feat
                msk_tau = m[:,0:idx+1,:,:]                              # Slice the pred mask
            else:
                emb_tau = cnn_fs[:,idx+1-self.tau:idx+1,:,:,:]
                msk_tau = m[:,idx+1-self.tau:idx+1,:,:]
            trans_tau, objaff_tau1 = self.self_attn1(emb_tau, msk_tau, num_cls)
            trans_tau, objaff_tau2 = self.self_attn2(trans_tau, msk_tau, num_cls)
            trans_tau, objaff_tau3 = self.self_attn3(trans_tau, msk_tau, num_cls)
            trans_tau, objaff_tau4 = self.self_attn4(trans_tau, msk_tau, num_cls)

            obj_prob_list = []
            for obj_id in range(0,num_cls):
                objaff_tau1_each = objaff_tau1[:,:,obj_id,:,:].unsqueeze(2)
                objaff_tau2_each = objaff_tau2[:,:,obj_id,:,:].unsqueeze(2)
                objaff_tau3_each = objaff_tau3[:,:,obj_id,:,:].unsqueeze(2)
                objaff_tau4_each = objaff_tau4[:,:,obj_id,:,:].unsqueeze(2)
                all_feat = torch.cat([trans_tau,emb_tau,objaff_tau1_each,objaff_tau2_each,objaff_tau3_each,objaff_tau4_each],dim=2).float()
                z = self.dec1(all_feat)
                z = self.dec2(z)
                z = self.dec3(z)
                obj_prob.append(z[-1])
                obj_prob = self.sigmoid(obj_prob)
                obj_prob_list.append(obj_prob)
            obj_prob = torch.cat(obj_prob_list,0)
            obj_pred = torch.argmax(obj_prob,0)

