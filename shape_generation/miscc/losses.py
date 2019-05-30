import torch
import torch.nn as nn

import sys
import numpy as np
from miscc.config import cfg
from miscc.utils import vgg_norm
from torch.autograd import Variable

# ##################Loss for G and Ds##############################
def ins_discriminator_loss(netINSD, real_hmaps, fake_hmaps, bbox_maps):
    # real_hmaps: batch x max_num_roi x 1 x hmap_size x hmap_size
    # fake_hmaps: batch x max_num_roi x 1 x hmap_size x hmap_size
    # bbox_maps: batch x max_num_roi x class_num x hmap_size x hmap_size

    batch_size, max_num_roi, hmap_size = real_hmaps.size(0), real_hmaps.size(1), real_hmaps.size(3)
    # prepare input
    real_input = torch.cat((real_hmaps, bbox_maps), 2)
    real_input = real_input.view(batch_size*max_num_roi, -1, hmap_size, hmap_size)
    fake_input = torch.cat((fake_hmaps.detach(), bbox_maps), 2)
    fake_input = fake_input.view(batch_size*max_num_roi, -1, hmap_size, hmap_size)
    # prepare labels
    real_labels = Variable(torch.FloatTensor(batch_size*max_num_roi).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size*max_num_roi).fill_(0))
    if cfg.CUDA:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()
    # Forward
    real_features = netINSD(real_input)
    fake_features = netINSD(fake_input)
    # loss
    #
    if len(cfg.GPU_IDS) > 1:
        real_logits = netINSD.module.get_logits(real_features)
    else:
        real_logits = netINSD.get_logits(real_features)
    real_logits = real_logits.squeeze()
    real_errD = nn.BCELoss()(real_logits, real_labels)
    if len(cfg.GPU_IDS) > 1:
        fake_logits = netINSD.module.get_logits(fake_features)
    else:
        fake_logits = netINSD.get_logits(fake_features)
    fake_logits = fake_logits.squeeze()
    fake_errD = nn.BCELoss()(fake_logits, fake_labels)

    errD = real_errD + fake_errD
    return errD

def glb_discriminator_loss(netGLBD, real_hmaps, fake_hmaps, bbox_maps):
    # real_hmaps: batch x 1 x hmap_size x hmap_size
    # fake_hmaps: batch x max_num_roi x 1 x hmap_size x hmap_size
    # bbox_maps: batch x max_num_roi x class_num x hmap_size x hmap_size

    fake_hmaps = fake_hmaps.detach()
    batch_size, max_num_roi, hmap_size = fake_hmaps.size(0), fake_hmaps.size(1), fake_hmaps.size(3)
    # prepare input
    # batch x max_num_roi x 1 x hmap_size x hmap_size -> batch x 1 x hmap_size x hmap_size
    fake_hmaps = torch.max(fake_hmaps, 1)[0]
    # batch x max_num_roi x class_num x hmap_size x hmap_size -> batch x class_num x hmap_size x hmap_size
    bbox_maps = torch.max(bbox_maps, 1)[0]

    real_input = torch.cat((real_hmaps, bbox_maps), 1)
    fake_input = torch.cat((fake_hmaps, bbox_maps), 1)
    # prepare labels
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    if cfg.CUDA:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()
    # Forward
    real_features = netGLBD(real_input)
    fake_features = netGLBD(fake_input)
    # loss
    #
    if len(cfg.GPU_IDS) > 1:
        real_logits = netGLBD.module.get_logits(real_features)
    else:
        real_logits = netGLBD.get_logits(real_features)
    real_logits = real_logits.squeeze()
    real_errD = nn.BCELoss()(real_logits, real_labels)
    if len(cfg.GPU_IDS) > 1:
        fake_logits = netGLBD.module.get_logits(fake_features)
    else:
        fake_logits = netGLBD.get_logits(fake_features)
    fake_logits = fake_logits.squeeze()
    fake_errD = nn.BCELoss()(fake_logits, fake_labels)

    errD = real_errD + fake_errD
    return errD


def generator_loss(netINSD, netGLBD, vgg_model, real_hmaps, fake_hmaps, bbox_maps):
    # real_hmaps: batch x max_num_roi x 1 x hmap_size x hmap_size
    # fake_hmaps: batch x max_num_roi x 1 x hmap_size x hmap_size
    # bbox_maps: batch x max_num_roi x class_num x hmap_size x hmap_size

    batch_size, max_num_roi, hmap_size = fake_hmaps.size(0), fake_hmaps.size(1), fake_hmaps.size(3)
    # prepare input
    fake_input = torch.cat((fake_hmaps, bbox_maps), 2)
    fake_input = fake_input.view(batch_size*max_num_roi, -1, hmap_size, hmap_size)
    # prepare labels
    real_labels = Variable(torch.FloatTensor(batch_size*max_num_roi).fill_(1))
    if cfg.CUDA:
        real_labels = real_labels.cuda()
    # Forward
    fake_features = netINSD(fake_input)
    # instance loss
    #
    if len(cfg.GPU_IDS) > 1:
        fake_logits = netINSD.module.get_logits(fake_features)
    else:
        fake_logits = netINSD.get_logits(fake_features)
    fake_logits = fake_logits.squeeze()
    insg_loss = nn.BCELoss()(fake_logits, real_labels)
    insg_loss *= cfg.TRAIN.SMOOTH.INS_LAMBDA

    # prepare input
    # batch x max_num_roi x 1 x hmap_size x hmap_size -> batch x 1 x hmap_size x hmap_size
    fake_pooled_hmaps = torch.max(fake_hmaps, 1)[0]
    # batch x max_num_roi x class_num x hmap_size x hmap_size -> batch x class_num x hmap_size x hmap_size
    bbox_maps = torch.max(bbox_maps, 1)[0]

    fake_input = torch.cat((fake_pooled_hmaps, bbox_maps), 1)
    # prepare labels
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    if cfg.CUDA:
        real_labels = real_labels.cuda()
    # Forward
    fake_features = netGLBD(fake_input)
    # global loss
    #
    if len(cfg.GPU_IDS) > 1:
        fake_logits = netGLBD.module.get_logits(fake_features)
    else:
        fake_logits = netGLBD.get_logits(fake_features)
    fake_logits = fake_logits.squeeze()
    glbg_loss = nn.BCELoss()(fake_logits, real_labels)
    glbg_loss *= cfg.TRAIN.SMOOTH.GLB_LAMBDA

    # prepare input
    # batch x max_num_roi x 1 x hmap_size x hmap_size -> batch x 1 x hmap_size x hmap_size
    real_pooled_hmaps = torch.max(real_hmaps, 1)[0]
    # batch x 1 x hmap_size x hmap_size -> batch x 3 x hmap_size x hmap_size
    real_hmaps = real_pooled_hmaps.repeat(1,3,1,1)
    fake_hmaps = fake_pooled_hmaps.repeat(1,3,1,1)

    norm_real_hmaps = vgg_norm(real_hmaps)
    norm_fake_hmaps = vgg_norm(fake_hmaps)

    # perceptual loss
    real_features = vgg_model(norm_real_hmaps)
    fake_features = vgg_model(norm_fake_hmaps)

    gpcp_loss = 0
    for i in range(len(real_features)):
        gpcp_loss += nn.L1Loss()(fake_features[i], real_features[i])
    item_pcp_score = gpcp_loss.item()
    gpcp_loss *= cfg.TRAIN.SMOOTH.PCP_LAMBDA

    errG_total = insg_loss + glbg_loss + gpcp_loss
    logs = 'insg_loss: %.2f glbg_loss: %.2f gpcp_loss: %.2f ' % (
        insg_loss.item(), glbg_loss.item(), gpcp_loss.item())

    return errG_total, logs, item_pcp_score