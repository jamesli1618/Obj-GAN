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
__C.RNN_TYPE = 'LSTM'   # 'GRU'

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 60
__C.TRAIN.SNAPSHOT_INTERVAL = 1
__C.TRAIN.PRINT_INTERVAL = 100
__C.TRAIN.DISPLAY_INTERVAL = 500
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.FLAG = True #True
__C.TRAIN.NET_E = '/pretrained/text_encoder100.pth'
__C.TRAIN.NET_G = ''
__C.TRAIN.BUATTN_NORM = True

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 4.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.DAMSM_LAMBDA = 100.0
__C.TRAIN.SMOOTH.TXT_LAMBDA = 0.1
__C.TRAIN.SMOOTH.SHP_LAMBDA = 1.0
__C.TRAIN.SMOOTH.OBJ_LAMBDA = 0.1
__C.TRAIN.SMOOTH.UNCOND_LAMBDA = 1.0

# Test options
__C.TEST = edict()
# 0 for gt box and gt seg 
# 1 for gt box and gen seg
# 2 for gen box and gen seg 
__C.TEST.USE_GT_BOX_SEG = 2 #0
__C.TEST.NET_SHP_G = '/pretrained/shape_ckpt/shape_gen.pth'
__C.TEST.SAVE_OPTIONS = 'IMAGE' # 'IMAGE' or 'SNAPSHOT'
__C.TEST.FID_DIMS = 2048 # Dimensionality of features returned by Inception
__C.TEST.USE_TF = 1
__C.TEST.TEST_IMG_NUM = 1000000
__C.TEST.RP_POOL_SIZE = 100
__C.TEST.SAMPLE_VAL = False

# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 96
__C.GAN.GF_DIM = 48
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.R_NUM = 1
__C.GAN.LOCAL_R_NUM = 3
__C.GAN.GLB_R_NUM = 7
__C.GAN.LAYER_D_NUM = 4

__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 5
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.GLOVE_EMBEDDING_DIM = 50 #[50, 100, 300]
__C.TEXT.WORDS_NUM = 12

# ROI options
__C.ROI = edict()
__C.ROI.BOXES_NUM = 10
__C.ROI.BOXES_DIM = 6 # left, top, width, height, category id, iscrowd
__C.ROI.FM_SIZE = 16
__C.ROI.ROI_MIN_SIZE = 10
__C.ROI.BOX_WORDS_NUM = 1
__C.ROI.ROI_BASE_SIZE = 5
__C.ROI.ROI_SIZE_THRS = 16.0