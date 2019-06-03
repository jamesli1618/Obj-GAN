import os
import argparse
import logging
import numpy as np
import math
import torch
from torch.optim.lr_scheduler import StepLR

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import PreEncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity, BBLoss
from seq2seq.optim import Optimizer
from seq2seq.dataset.prepare_dataset import prepare_data, prepare_test_data, \
    random_batch, get_class_sta
from seq2seq.evaluator import Evaluator
from seq2seq.util.checkpoint import Checkpoint

import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path', 
    default='../data/coco/box_label/input_train2014.txt',
    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path', 
    default='../data/coco/box_label/input_val2014.txt',
    help='Path to dev data')
parser.add_argument('--train_filename_path', action='store', dest='train_filename_path', 
    default='../data/coco/box_label/filenames_train2014.txt',
    help='Path to train filename data')
parser.add_argument('--dev_filename_path', action='store', dest='dev_filename_path', 
    default='../data/coco/box_label/filenames_val2014.txt',
    help='Path to dev filename data')
parser.add_argument('--mean_std_path', action='store', dest='mean_std_path', 
    default='../data/coco/box_label/mean_std_train2014.txt',
    help='Path to dev data')
parser.add_argument('--gaussian_dict_path', action='store', dest='gaussian_dict_path', 
    default='../data/coco/box_label/gaussian_dict.npy',
    help='Path to gaussian dict')
parser.add_argument('--vocab_path', action='store', dest='vocab_path', 
    default='../data/coco/captions.pickle',
    help='Path to the vocab path')
parser.add_argument('--box_saving_folder', action='store', dest='box_saving_folder', 
    default='../data/coco/gen_masks',
    help='Path to box saving folder')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='experiment',
    help='Path to experiment directory. If load_checkpoint is True, \
    then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint', 
    default='../data/pretrained/coco/box_ckpt', help='The name of the checkpoint to load, \
    usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume', default=False, 
    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level', default='info', help='Logging level.')
parser.add_argument('--batch_size', type=int, default=1, help='The batch size.')
parser.add_argument('--gmm_comp_num', type=int, default=5, help='The number of GMM components.')
parser.add_argument('--lamda1', type=float, default=1.0, 
    help='The balancing parameter for label loss.')
parser.add_argument('--lamda2', type=float, default=1.0, 
    help='The balancing parameter for box loss.')
parser.add_argument('--count_smooth', type=float, default=1e5, 
    help='The balancing parameter for box loss.')
parser.add_argument('--is_training', type=int, default=1, help='The state for training or test')
parser.add_argument('--max_len', type=int, default=150, help='The max length for sequences')
parser.add_argument('--min_len', type=int, default=1, help='The min length for sequences')
parser.add_argument('--early_stop_len', type=int, default=10, 
    help='The early-stop length for generation')
parser.add_argument('--output_opt', type=int, default=0, help='The output option (0/1)')
parser.add_argument('--embedding_dim', type=int, default=256, help='The embedding dimension')
parser.add_argument('--encoder_path', type=str, 
    default='../data/coco/pretrained/text_encoder100.pth', 
    help='encoder path.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.is_training:
    # load vocab
    x = pickle.load(open(opt.vocab_path, 'rb'))
    ixtoword, wordtoix = x[2], x[3]
    del x

    # prepare dataset
    train_cap_lang, train_label_lang, train_tuples, dev_cap_lang, dev_label_lang, dev_tuples, \
        x_mean_std, y_mean_std, w_mean_std, r_mean_std = prepare_data(opt.train_path, 
            opt.dev_path, opt.mean_std_path, opt.max_len, opt.min_len, ixtoword, wordtoix)

    weight = torch.ones(len(train_label_lang.word2index))
    for word in train_label_lang.word2index:
        if train_label_lang.word2count[word] == 0:
            continue
        index = train_label_lang.word2index[word]
        weight[index] = weight[index]*opt.count_smooth/float(math.pow(
        	train_label_lang.word2count[word], 0.8))

    # Prepare loss
    pad = train_label_lang.word2index["<pad>"]
    lloss = Perplexity(weight, pad, opt.lamda1)
    bloss = BBLoss(opt.batch_size, opt.gmm_comp_num, opt.lamda2)
    if torch.cuda.is_available():
        lloss.cuda()
        bloss.cuda()

    print('train_label_lang.index2word:')
    for index in train_label_lang.index2word:
        print('{} : {} '.format(index, train_label_lang.index2word[index]))
     
    print('train_label_lang.word2count:')
    for word in train_label_lang.word2count:
        print('{} : {} '.format(word, train_label_lang.word2count[word]))

    hidden_size = opt.embedding_dim
    encoder = PreEncoderRNN(train_cap_lang.n_words, nhidden=opt.embedding_dim)
    state_dict = torch.load(opt.encoder_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(state_dict)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    if torch.cuda.is_available():
        encoder.cuda()

    seq2seq = None
    optimizer = None
    bidirectional = True
    if not opt.resume:
        # Initialize model
        decoder = DecoderRNN(train_label_lang.word2index, x_mean_std[0], y_mean_std[0], 
            w_mean_std[0], r_mean_std[0], opt.batch_size, opt.max_len, hidden_size, 
            opt.gmm_comp_num, dropout_p=0.2, use_attention=False, bidirectional=bidirectional)

        for param in decoder.parameters():
            param.data.uniform_(-0.08, 0.08)

        if torch.cuda.is_available():
            decoder.cuda()

    # train
    t = SupervisedTrainer(lloss=lloss, bloss=bloss, batch_size=opt.batch_size, 
        checkpoint_every=100, print_every=50, expt_dir=opt.expt_dir, 
        train_cap_lang=train_cap_lang, train_label_lang=train_label_lang, x_mean_std=x_mean_std, 
        y_mean_std=y_mean_std, w_mean_std=w_mean_std, r_mean_std=r_mean_std)

    seq2seq = t.train(encoder, decoder, train_tuples, num_epochs=10, dev_data=dev_tuples, 
        optimizer=optimizer, resume=opt.resume, is_training=opt.is_training)

elif not opt.is_training and opt.load_checkpoint is not None:
    opt.box_saving_folder = '%s_%s/'%(opt.box_saving_folder, opt.load_checkpoint)

    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, 
        Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, 
        opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    decoder = checkpoint.model
    decoder.eval()

    cap_word2index = checkpoint.cap_word2index
    cap_index2word = checkpoint.cap_index2word
    label_word2index = checkpoint.label_word2index
    label_index2word = checkpoint.label_index2word

    if not os.path.isfile(opt.gaussian_dict_path):
        print('calculating means and stds of box positions and sizes...')
        get_class_sta(opt.train_path, opt.gaussian_dict_path)

    gaussian_dict = np.load(opt.gaussian_dict_path).item()

    hidden_size = opt.embedding_dim
    encoder = PreEncoderRNN(len(cap_word2index), nhidden=opt.embedding_dim)
    state_dict = torch.load(opt.encoder_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(state_dict)
    encoder.eval()

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # prepare dataset
    dev_cap_lang, dev_label_lang, dev_tuples, x_mean_std, y_mean_std, w_mean_std, r_mean_std, \
    keys = prepare_test_data(opt.dev_path, opt.mean_std_path, opt.max_len, opt.min_len, 
        cap_word2index, cap_index2word, label_word2index, label_index2word, opt.dev_filename_path)

    evaluator = Evaluator(opt.batch_size, opt.early_stop_len, opt.expt_dir, dev_cap_lang, 
        dev_label_lang, x_mean_std, y_mean_std, w_mean_std, r_mean_std, gaussian_dict, 
        opt.box_saving_folder, opt.output_opt)
    evaluator.evaluate(encoder, decoder, dev_tuples, keys)
