from __future__ import print_function

from miscc.config import cfg
from miscc.load import load_acts_data
from trainDataset import TrainDataset
from testDataset import TestDataset
from trainer import condGANTrainer as trainer
from evaluator import condGANEvaluator as evaluator

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Train an image generation network')
    parser.add_argument('--gpu', dest='gpu_ids', type=str, default='-1')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='../data/coco')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    # output dir
    parser.add_argument('--output_dir', type=str, default='..')
    parser.add_argument('--MAX_EPOCH', type=int, default=60)
    parser.add_argument('--WORKERS', type=int, default=0)
    parser.add_argument('--NET_G', type=str, default='') # 
    parser.add_argument('--SAMPLE_VAL', dest='SAMPLE_VAL', action='store_true')
    parser.add_argument('--PRINT_INTERVAL', type=int, default=100)
    parser.add_argument('--DISPLAY_INTERVAL', type=int, default=500)
    # tunable argument
    parser.add_argument('--DISCRIMINATOR_LR', type=float, default=0.0002)
    parser.add_argument('--GENERATOR_LR', type=float, default=0.0002)
    parser.add_argument('--DAMSM_LAMBDA', type=float, default=100.0)
    parser.add_argument('--TXT_LAMBDA', type=float, default=0.1)
    parser.add_argument('--SHP_LAMBDA', type=float, default=1.0)
    parser.add_argument('--OBJ_LAMBDA', type=float, default=0.1)
    parser.add_argument('--UNCOND_LAMBDA', type=float, default=1.0)
    parser.add_argument('--GLB_R_NUM', type=int, default=7)
    parser.add_argument('--LAYER_D_NUM', type=int, default=4)
    parser.add_argument('--BATCH_SIZE', type=int, default=24)
    parser.add_argument('--BRANCH_NUM', type=int, default=3)
    parser.add_argument('--FLAG', dest='FLAG', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    cfg.TRAIN.NET_G = args.NET_G
    cfg.TRAIN.NET_E = cfg.DATA_DIR + cfg.TRAIN.NET_E
    cfg.TEST.NET_SHP_G = cfg.DATA_DIR + cfg.TEST.NET_SHP_G
    cfg.TRAIN.MAX_EPOCH = args.MAX_EPOCH
    cfg.WORKERS = args.WORKERS
    cfg.TEST.SAMPLE_VAL = args.SAMPLE_VAL
    cfg.TRAIN.PRINT_INTERVAL = args.PRINT_INTERVAL
    cfg.TRAIN.DISPLAY_INTERVAL = args.DISPLAY_INTERVAL
    #
    cfg.TRAIN.DISCRIMINATOR_LR = args.DISCRIMINATOR_LR
    cfg.TRAIN.GENERATOR_LR = args.GENERATOR_LR
    cfg.TRAIN.SMOOTH.DAMSM_LAMBDA = args.DAMSM_LAMBDA
    cfg.TRAIN.SMOOTH.TXT_LAMBDA = args.TXT_LAMBDA
    cfg.TRAIN.SMOOTH.SHP_LAMBDA = args.SHP_LAMBDA
    cfg.TRAIN.SMOOTH.OBJ_LAMBDA = args.OBJ_LAMBDA
    cfg.TRAIN.SMOOTH.UNCOND_LAMBDA = args.UNCOND_LAMBDA
    cfg.GAN.GLB_R_NUM = args.GLB_R_NUM
    cfg.GAN.LAYER_D_NUM = args.LAYER_D_NUM
    cfg.TRAIN.BATCH_SIZE = args.BATCH_SIZE
    cfg.TREE.BRANCH_NUM = args.BRANCH_NUM
    cfg.TRAIN.FLAG = args.FLAG

    if args.gpu_ids != '-1':
        cfg.GPU_IDS = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    else:
        cfg.CUDA = False

    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '{0}/output_image_generation/{1}_{2}'.format(args.output_dir, 
        cfg.DATASET_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    start_t = time.time()
    if cfg.TRAIN.FLAG:
        dataset = TrainDataset(cfg.DATA_DIR, split_dir, base_size=cfg.TREE.BASE_SIZE)
        assert dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

        # Define models and go to training
        algo = trainer(output_dir, dataloader, dataset)
        algo.train()
    else:
        dataset = TestDataset(cfg.DATA_DIR, split_dir, base_size=cfg.TREE.BASE_SIZE)
        assert dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))
        
        # Define models and go to evaluating
        algo = evaluator(output_dir, dataloader, dataset)

        if dataset.acts_dict is None:
            algo.dump_fid_acts(cfg.DATA_DIR, split_dir)
            dataset.acts_dict = load_acts_data(cfg.DATA_DIR, split_dir)
        
        algo.evaluate(split_dir, dataset.imsize)
    end_t = time.time()
    print('Total time for {0}:'.format(split_dir), end_t - start_t)