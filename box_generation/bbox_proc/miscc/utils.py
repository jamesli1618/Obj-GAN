import os
import errno
import numpy as np
from torch.nn import init

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import skimage.transform
import ntpath
import sys

from miscc.config import cfg


# For visualization ################################################
COLOR_DIC = {0:[128,64,128],  1:[244, 35,232],
             2:[70, 70, 70],  3:[102,102,156],
             4:[190,153,153], 5:[153,153,153],
             6:[250,170, 30], 7:[220, 220, 0],
             8:[107,142, 35], 9:[152,251,152],
             10:[70,130,180], 11:[220,20, 60],
             12:[255, 0, 0],  13:[0, 0, 142],
             14:[119,11, 32], 15:[0, 60,100],
             16:[0, 80, 100], 17:[0, 0, 230],
             18:[0,  0, 70],  19:[0, 0,  0]}
FONT_MAX = 50


def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    # get a font
    # fnt = None  # ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    fnt = ImageFont.truetype('../share/Pillow/Tests/fonts/FreeMono.ttf', 50)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list


def build_super_images(real_imgs, captions, ixtoword,
                       attn_maps, att_sze, lr_imgs=None,
                       batch_size=cfg.TRAIN.BATCH_SIZE,
                       max_word_num=cfg.TEXT.WORDS_NUM):
    nvis = 8
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]
    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2)

    text_convas = \
        np.ones([batch_size * FONT_MAX,
                 (max_word_num + 2) * (vis_size + 2), 3],
                dtype=np.uint8)

    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]


    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
    if lr_imgs is not None:
        lr_imgs = \
            nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # --> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)
        
        attn = torch.cat([attn_max[0], attn], 1)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]
        #
        img = real_imgs[i]
        if lr_imgs is None:
            lrI = img
        else:
            lrI = lr_imgs[i]
        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]
        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            one_map = attn[j]
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze)
            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
        for j in range(seq_len + 1):
            if j < num_attn:
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                one_map *= 255
                #
                PIL_im = Image.fromarray(np.uint8(img))
                PIL_att = Image.fromarray(np.uint8(one_map))
                merged = \
                    Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new('L', (vis_size, vis_size), (210))
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)
                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad
            row.append(one_map)
            row.append(middle_pad)
            #
            row_merge.append(merged)
            row_merge.append(middle_pad)
        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


def build_super_images2(real_imgs, captions, ixtoword,
                       attn_maps, att_sze, lr_imgs=None,
                       batch_size=cfg.TRAIN.BATCH_SIZE,
                       max_word_num=cfg.TEXT.WORDS_NUM):
    nvis = 8
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]
    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2)

    text_convas = \
        np.ones([batch_size * FONT_MAX,
                 (max_word_num + 2) * (vis_size + 2), 3],
                dtype=np.uint8)

    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]


    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
    if lr_imgs is not None:
        lr_imgs = \
            nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # --> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)
        
        attn = torch.cat([attn_max[0], attn], 1)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]
        #
        img = real_imgs[i]
        if lr_imgs is None:
            lrI = img
        else:
            lrI = lr_imgs[i]
        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]
        row_beforeNorm = []
        minVglobal, maxVglobal = 0.0, 0.75
        for j in range(num_attn):
            one_map = attn[j]
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze)
            row_beforeNorm.append(one_map)
            '''minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV'''
        for j in range(seq_len + 1):
            if j < num_attn:
                one_map = row_beforeNorm[j]
                one_map = np.clip(one_map, minVglobal, maxVglobal)
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                one_map *= 255
                #
                PIL_im = Image.fromarray(np.uint8(img))
                PIL_att = Image.fromarray(np.uint8(one_map))
                merged = \
                    Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new('L', (vis_size, vis_size), (210))
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)
                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad
            row.append(one_map)
            row.append(middle_pad)
            #
            row_merge.append(merged)
            row_merge.append(middle_pad)
        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


####################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def path_leaf(path):
    return ntpath.basename(path)

def read_hmap(hmap_path):
    hmap = Image.open(hmap_path)
    hmap = np.asarray(hmap)
    hmap = np.squeeze(hmap[:,:,0])
    return hmap

def crop_rois(roi_cnn_model, fmaps, fm_rois, num_rois, roi_size, nef):
    # input:
    # fmaps (type = variable): batch_size x nef x H x W
    # rois (type = numpy): batch_size x cfg.ROI.BOXES_NUM x cfg.ROI.BOXES_DIM (left, top, width, height, category id, iscrowd)
    # num_rois (type = list): batch_size
    # output:
    # cropped_rois (variable): batch_size x nef x max_num_rois x 1
    num_rois = num_rois.cpu().data.numpy().tolist()
    fmap_size = int(fmaps.size(2))
    max_num_roi = np.amax(num_rois)
    cropped_rois = []
    batch_size = len(num_rois)
    cropped_rois = Variable(torch.FloatTensor(batch_size, nef, max_num_roi, 1).zero_())
    if cfg.CUDA:
        cropped_rois = cropped_rois.cuda()
    for batch_index in xrange(batch_size):
        if num_rois[batch_index] == 0:
            continue
        # num_rois[batch_index] x nef x roi_size x roi_size
        rois_within_batch = Variable(torch.FloatTensor(num_rois[batch_index], nef, roi_size, roi_size))
        if cfg.CUDA:
            rois_within_batch = rois_within_batch.cuda()
        for roi_index in xrange(num_rois[batch_index]):
            left, top, width, height = fm_rois[batch_index, roi_index, :4]
            left, top, width, height = int(round(left)), int(round(top)), int(round(width)), int(round(height))

            # in left or top is on the border
            left = min(left, roi_size-1)
            top = min(top, roi_size-1)
            # in case width or height is zero
            width = max(width, 1)
            height = max(height, 1)
            width = min(width, fmap_size-left)
            height = min(height, fmap_size-top)
            roi_fmap = fmaps[batch_index, :, top:top+height, left:left+width].clone()
            # roi_fmap: 1 x C x roi_size x roi_size
            roi_fmap = nn.AdaptiveAvgPool2d((roi_size, roi_size))(roi_fmap)
            rois_within_batch[roi_index] = roi_fmap

        # num_rois[batch_index] x nef x 1 x 1
        rois_within_batch = roi_cnn_model(rois_within_batch, roi_size)
        rois_within_batch = rois_within_batch.squeeze()
        #print('rois_within_batch.size()1: ', rois_within_batch.size())
        #print('rois_within_batch.dim(): ', rois_within_batch.dim())
        if rois_within_batch.dim() == 1:
            rois_within_batch = rois_within_batch.unsqueeze(0)
        #print('rois_within_batch.size()2: ', rois_within_batch.size())
        rois_within_batch = rois_within_batch.transpose(1, 0)
        #print('rois_within_batch.size()3: ', rois_within_batch.size())
        cropped_rois[batch_index,:,:num_rois[batch_index],0] = rois_within_batch

    return cropped_rois

def pprocess_bt_attns(bt_attn_maps, rois, num_rois, img_size):
    # attn_maps[0]: 1 x num_rois x words_num
    num_rois = num_rois.cpu().data.numpy().tolist()
    new_bt_attn_maps = []

    for batch_index in xrange(len(num_rois)):
        attn_map = bt_attn_maps[batch_index]
        num_attn = attn_map.size(2) # words_num
        tmp_num_rois = max(num_rois[batch_index], 1)
        stacked_new_one_maps = Variable(torch.FloatTensor(num_attn, tmp_num_rois, img_size, img_size).zero_())
        if cfg.CUDA:
            stacked_new_one_maps = stacked_new_one_maps.cuda()

        if num_rois[batch_index] > 0:
            one_map = attn_map[0]
            #print('one_map.size(): ', one_map.size())
            #print('num_rois[batch_index]: ', num_rois[batch_index])
            for roi_index in xrange(num_rois[batch_index]):
                # get the attention weight for the roi_index-th region
                val = one_map[roi_index]

                # get the current roi
                left, top, width, height = rois[batch_index, roi_index, :4]
                left, top, width, height = int(round(left)), int(round(top)), int(round(width)), int(round(height))

                # in case width or height is zero
                width = max(width, 1)
                height = max(height, 1)

                # in case width or height exceeds the img_size
                width = min(width, img_size-left)
                height = min(height, img_size-top)

                all_ones = Variable(torch.from_numpy(np.ones((num_attn, height, width))).float())
                if cfg.CUDA:
                    all_ones = all_ones.cuda()
                for attn_index in xrange(num_attn):
                    all_ones[attn_index] = all_ones[attn_index]*val[attn_index]

                # project the attention weight back to the image plane according to the roi coordinates
                stacked_new_one_maps[:, roi_index, top:top+height, left:left+width] = all_ones

        # do the max pooling for the stacked_new_one_maps along Dim 0
        new_attn_map = torch.max(stacked_new_one_maps, dim=1)[0].unsqueeze(0)
        new_bt_attn_maps.append(new_attn_map)
            
    return new_bt_attn_maps

def reorg_attns(attn_maps, cap_filter_masks, words_emb):
    # attn_maps[0]: 1 x words_num x H x W
    cap_filter_masks = cap_filter_masks[:,0:words_emb.shape[2]]
    new_attn_maps = []
    for i in xrange(len(attn_maps)):
        mask = cap_filter_masks[i]
        indices = np.where(mask == 0)[0].tolist()
        attn = attn_maps[i].cpu().data.numpy()
        new_attn_map = np.zeros((attn.shape[0], len(mask), attn.shape[2], attn.shape[3]))
        for j in xrange(len(indices)):
            new_attn_map[0,indices[j],:,:] = attn[0,j,:,:]

        if cfg.CUDA:
            new_attn_maps.append(Variable(torch.from_numpy(new_attn_map).float()).cuda())
        else:
            new_attn_maps.append(Variable(torch.from_numpy(new_attn_map).float()))

    return new_attn_maps

def form_uni_batch(bbox_labels, bbox_label_lens, num_rois, batch_size):
    # input:
    # bbox_labels (type = numpy array): batch_size x max_num_rois
    # bbox_label_lens (type = numpy array): batch_size x max_num_rois
    # num_rois (type = list): batch_size
    # output:
    # uni_bbox_labels (type = variable): uni_batch_size
    # uni_bbox_label_lens (type = variable): uni_batch_size
    num_rois = num_rois.cpu().data.numpy().tolist()
    bbox_labels = bbox_labels.cpu().data.numpy()
    uni_bbox_labels = []
    uni_bbox_label_lens = []
    for batch_index in xrange(batch_size):
        if num_rois[batch_index] == 0:
            continue
        bbox_labels_within_batch = bbox_labels[batch_index, :num_rois[batch_index]]
        bbox_label_lens_within_batch = bbox_label_lens[batch_index, 0:num_rois[batch_index]]
        uni_bbox_labels.append(bbox_labels_within_batch)
        uni_bbox_label_lens.append(bbox_label_lens_within_batch)

    uni_bbox_labels = np.concatenate(uni_bbox_labels, axis=0)
    uni_bbox_label_lens = np.concatenate(uni_bbox_label_lens, axis=0)    
    uni_batch_size = len(uni_bbox_label_lens)

    uni_bbox_labels = torch.from_numpy(uni_bbox_labels).unsqueeze(1)
    uni_bbox_label_lens = torch.from_numpy(uni_bbox_label_lens)

    if cfg.CUDA:
        uni_bbox_labels = Variable(uni_bbox_labels).cuda()
        uni_bbox_label_lens = Variable(uni_bbox_label_lens).cuda()
    else:
        uni_bbox_labels = Variable(uni_bbox_labels)
        uni_bbox_label_lens = Variable(uni_bbox_label_lens)

    return uni_bbox_labels, uni_bbox_label_lens, uni_batch_size

def reorg_bbox_label_feats(uni_bbox_sent_emb, num_rois):
    # input:
    # uni_bbox_sent_emb (type = variable): uni_batch_size x nef
    # num_rois (type = list): batch_size
    # output:
    # bbox_label_emb (type = variable): batch_size x nef x max_num_roi
    num_rois = num_rois.cpu().data.numpy().tolist()
    batch_size = len(num_rois)
    nef = uni_bbox_sent_emb.size(1)
    max_num_roi = np.amax(num_rois)

    bbox_label_emb = Variable(torch.FloatTensor(batch_size, nef, max_num_roi).zero_())
    if cfg.CUDA:
        bbox_label_emb = bbox_label_emb.cuda()

    dynamic_start_index = 0
    for batch_index in xrange(len(num_rois)):
        if num_rois[batch_index] == 0:
            continue
        tmp_indices = Variable(torch.LongTensor(range(dynamic_start_index, \
            dynamic_start_index+num_rois[batch_index])))
        if cfg.CUDA:
            tmp_indices = tmp_indices.cuda()

        bbox_label_emb[batch_index, :, :num_rois[batch_index]] = \
            uni_bbox_sent_emb.index_select(0, tmp_indices).transpose(0,1)
        dynamic_start_index += num_rois[batch_index]

    return bbox_label_emb

def form_uni_cap_batch(captions, seq_len, cap_lens, cap_filter_masks):
    # captions (type = variable): batch_size x cfg.TEXT.WORDS_NUM
    # cap_lens (type = variable): batch_size
    # cap_filter_masks (type = numpy): batch x cfg.TEXT.WORDS_NUM
   
    cap_lens = cap_lens.cpu().data.numpy()
    cap_filter_masks = cap_filter_masks[:,:seq_len]

    uni_filtered_captions = []
    all_keep_indices = []
    for batch_index in xrange(len(cap_lens)):
        keep_indices = Variable(torch.from_numpy(np.where(cap_filter_masks[batch_index,:] == 0)[0]))
        if cfg.CUDA:
            keep_indices = keep_indices.cuda()
        #print('keep_indices: ')
        #print(keep_indices)
        all_keep_indices.append(keep_indices)
        new_flt_cap = captions[batch_index,:].index_select(0, keep_indices)
        new_flt_cap = new_flt_cap.view(len(keep_indices), 1)
        if batch_index == 0:
            uni_filtered_captions = new_flt_cap
        else:
            uni_filtered_captions = torch.cat([uni_filtered_captions, new_flt_cap], dim=0)

    uni_filtered_cap_lens = Variable(torch.from_numpy(np.ones(uni_filtered_captions.size(0))).long())

    if cfg.CUDA:
        uni_filtered_cap_lens = uni_filtered_cap_lens.cuda()

    uni_cap_batch_size = uni_filtered_captions.size(0)

    return uni_filtered_captions, uni_filtered_cap_lens, uni_cap_batch_size


def reorg_flt_words_feats(uni_flt_words_emb, words_emb, captions, cap_lens, cap_filter_masks):
    cap_lens = cap_lens.cpu().data.numpy()
    filtered_cap_lens = np.sum(cap_filter_masks, axis=1)
    filtered_cap_lens = np.subtract(np.ones(cap_lens.shape)*cap_filter_masks.shape[1], filtered_cap_lens)
    filtered_captions = Variable(torch.FloatTensor(captions.size(0), \
        int(np.max(filtered_cap_lens))).zero_())
    filtered_words_emb = Variable(torch.FloatTensor(words_emb.size(0), \
        words_emb.size(1), int(np.max(filtered_cap_lens))).zero_())
    if cfg.CUDA:
        filtered_captions = filtered_captions.cuda()
        filtered_words_emb = filtered_words_emb.cuda()

    filtered_cap_lens = np.zeros(cap_lens.shape, dtype='int64')

    accum_index = 0
    for batch_index in xrange(len(cap_lens)):
        keep_indices = Variable(torch.from_numpy(np.where(cap_filter_masks[batch_index,:] == 0)[0]))
        uni_indices = Variable(torch.from_numpy(np.asarray(range(accum_index, accum_index+len(keep_indices)))))
        if cfg.CUDA:
            keep_indices = keep_indices.cuda()
            uni_indices = uni_indices.cuda()

        filtered_words_emb[batch_index,:,:len(keep_indices)] = uni_flt_words_emb.index_select(0, uni_indices).transpose(0,1)
        filtered_cap_lens[batch_index] = len(keep_indices)
        filtered_captions[batch_index,:len(keep_indices)] = captions[batch_index,:].index_select(0, keep_indices)
        accum_index += len(keep_indices)

    filtered_cap_lens = Variable(torch.from_numpy(filtered_cap_lens))
    if cfg.CUDA:
        filtered_cap_lens = filtered_cap_lens.cuda()

    return filtered_words_emb, filtered_captions, filtered_cap_lens