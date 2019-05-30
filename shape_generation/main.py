from __future__ import print_function

from miscc.config import cfg
from datasets import TextDataset
from trainer import condGANTrainer as trainer

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
    parser = argparse.ArgumentParser(description='Train a shape generation network')
    parser.add_argument('--gpu', dest='gpu_ids', type=str, default='-1')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='../data/coco')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--output_dir', type=str, default='..')
    parser.add_argument('--MAX_EPOCH', type=int, default=60)
    parser.add_argument('--WORKERS', type=int, default=0)
    parser.add_argument('--NET_G', type=str, default='')
    # tunable argument
    parser.add_argument('--DISCRIMINATOR_LR', type=float, default=0.0002)
    parser.add_argument('--GENERATOR_LR', type=float, default=0.0002)
    parser.add_argument('--INS_LAMBDA', type=float, default=1.0)
    parser.add_argument('--GLB_LAMBDA', type=float, default=1.0)
    parser.add_argument('--PCP_LAMBDA', type=float, default=10.0)
    parser.add_argument('--R_NUM', type=int, default=1)
    parser.add_argument('--BATCH_SIZE', type=int, default=40)
    parser.add_argument('--FLAG', dest='FLAG', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    cfg.TRAIN.NET_G = args.NET_G
    cfg.TRAIN.NET_E = cfg.DATA_DIR + cfg.TRAIN.NET_E
    cfg.TRAIN.MAX_EPOCH = args.MAX_EPOCH
    cfg.TRAIN.WORKERS = args.WORKERS
    #
    cfg.TRAIN.DISCRIMINATOR_LR = args.DISCRIMINATOR_LR
    cfg.TRAIN.GENERATOR_LR = args.GENERATOR_LR
    cfg.TRAIN.SMOOTH.INS_LAMBDA = args.INS_LAMBDA
    cfg.TRAIN.SMOOTH.GLB_LAMBDA = args.GLB_LAMBDA
    cfg.TRAIN.SMOOTH.PCP_LAMBDA = args.PCP_LAMBDA
    cfg.GAN.R_NUM = args.R_NUM
    cfg.TRAIN.BATCH_SIZE = args.BATCH_SIZE
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
    output_dir = '{0}/output_shape_gen/{1}_{2}'.format(args.output_dir, cfg.DATASET_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        bshuffle = False
        split_dir = 'test'

    # Get data loader
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.sampling(split_dir)  # generate images for the whole val dataset

    end_t = time.time()
    print('Total time for training:', end_t - start_t)
