from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'coco'
__C.DATA_DIR = ''
__C.GPU_IDS = '0'
__C.CUDA = True
__C.WORKERS = 0

__C.RNN_TYPE = 'LSTM'

__C.TREE = edict()
__C.TREE.BASE_SIZE = 64

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 44
__C.TRAIN.MAX_EPOCH = 60
__C.TRAIN.SNAPSHOT_INTERVAL = 1
__C.TRAIN.PRINT_INTERVAL = 5 #100
__C.TRAIN.DISPLAY_INTERVAL = 5 #500
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = '/pretrained/text_encoder100.pth'
__C.TRAIN.NET_G = ''

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.INS_LAMBDA = 1.0
__C.TRAIN.SMOOTH.GLB_LAMBDA = 1.0
__C.TRAIN.SMOOTH.PCP_LAMBDA = 10.0

# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 96
__C.GAN.GF_DIM = 96
__C.GAN.Z_DIM = 100
__C.GAN.R_NUM = 1

# Text options
__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 5
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 40

# ROI options
__C.ROI = edict()
__C.ROI.BOXES_NUM = 10
__C.ROI.BOXES_DIM = 6 # left, top, width, height, category id, iscrowd
__C.ROI.FM_SIZE = 16
__C.ROI.ROI_MIN_SIZE = 2

__C.VGG = edict()
# Pixel mean values (RGB order) as a (1, 1, 3) array
__C.VGG.PIXEL_MEANS = np.array([[[0.485, 0.456, 0.406]]])
__C.VGG.PIXEL_STDS = np.array([[[0.229, 0.224, 0.225]]])