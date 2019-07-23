import os
import errno
import numpy as np
from torch.nn import init

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import skimage.transform
import ntpath
import random
from scipy import linalg

from miscc.config import cfg
from models.roi_align.modules.roi_align import RoIAlignAvg


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
    fnt = ImageFont.truetype(cfg.DATA_DIR + '/share/Pillow/Tests/fonts/FreeMono.ttf', 50)
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

def build_super_shape_images(real_imgs, captions, ixtoword,
                       attn_maps, att_sze, lr_imgs=None,
                       font_max = 50, font_size = 50,
                       batch_size=cfg.TRAIN.BATCH_SIZE,
                       max_word_num=cfg.TEXT.WORDS_NUM):
    nvis = min(8, batch_size)
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]
    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2)

    text_convas = \
        np.ones([batch_size * font_max,
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
        drawCaption(text_convas, captions, ixtoword, vis_size, font_max, font_size)
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
        for j in range(seq_len + 1):
            if j < num_attn:
                one_map = row_beforeNorm[j]
                minV = one_map.min()
                maxV = one_map.max()
                if maxV != minV:
                    one_map = (one_map - minV) / (maxV - minV)
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
        txt = text_map[i * font_max: (i + 1) * font_max]
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
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
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

def is_non_zero_file(fpath):  
    return True if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else False

def calc_sort_size(boxes_arr, w_index=2, h_index=3):
    # boxes_arr (type = numpy array): boxes_num x 6 (x, y, w, h, l, crowd_l)
    # calculate the product of width and height
    sizes = np.multiply(boxes_arr[:, w_index], boxes_arr[:, h_index])

    # sort sizes in the ascending order
    sorted_indices = np.argsort(sizes)[::-1].tolist()
    sorted_boxes_arr = boxes_arr[sorted_indices,:]

    return sorted_boxes_arr, sorted_indices

def denorm_imgs(images):
    denorm_images = images.add(1).div(2).mul(255).clamp(0, 255).byte()
    denorm_images = denorm_images.data.cpu().numpy()

    return denorm_images

############################################################################
def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    levels = np.arange(im_rois.shape[0]//cfg.ROI.BOXES_NUM).astype(np.int)
    levels = np.expand_dims(levels, axis=1)
    levels = np.repeat(levels, cfg.ROI.BOXES_NUM, axis=1)
    levels = np.reshape(levels, (levels.shape[0]*levels.shape[1], 1))

    rois = im_rois * scales[levels]

    return rois, levels

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def pprocess_bt_attns(fmaps, ih, iw, bt_mask):
    # fmaps: batch x num (idf or cap_len) x max_num_rois x 1
    batch_size, num, max_num_rois = fmaps.size(0), fmaps.size(1), fmaps.size(2)
    # fmaps: batch x num x max_num_rois x ih x iw
    fmaps = fmaps.repeat(1, 1, 1, ih*iw).view(batch_size, -1, max_num_rois, ih, iw)
    # fmaps: batch x max_num_rois x num x ih x iw
    fmaps = fmaps.transpose(1, 2)
    # bt_mask: batch x max_num_rois x num x ih x iw
    fmaps = fmaps * bt_mask
    # fmaps: batch x num x ih x iw
    fmaps = torch.max(fmaps, dim=1)[0]

    return fmaps


####################################################################
def compute_inception_score(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def negative_log_posterior_probability(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)


####################################################################
def permute_seg(seg_conditions, rois, num_rois):
    new_seg_conditions = seg_conditions.clone()
    batch_size = seg_conditions.size(0)
    valid_mask = []
    for batch_index in range(batch_size):
        if num_rois[batch_index] == 0:
            continue
        classes = rois[batch_index,:num_rois[batch_index],4]
        unique_classes = list(np.unique(classes).astype(np.int))
        rand_unique_classes = deepcopy(unique_classes)
        random.shuffle(rand_unique_classes)

        if unique_classes != rand_unique_classes:
            valid_mask.append(batch_index)
            new_seg_conditions[batch_index, unique_classes] = \
                seg_conditions[batch_index, rand_unique_classes]

    return new_seg_conditions, valid_mask


def feat_select(pooled_feat, raw_bt_c_codes, fm_rois, num_rois, is_large_scale=False):
    fm_rois = fm_rois.data.cpu().numpy()
    x_code_rois, classes, bt_c_codes = [], [], []
    batch_size = len(num_rois)
    for batch_index in range(batch_size):
        if num_rois[batch_index] == 0:
            continue

        real_indices = []
        for roi_index in range(num_rois[batch_index]):
            left, top, width, height = fm_rois[batch_index, roi_index, :4]
            if width < 1.25 and height < 1.25:
                continue
            if is_large_scale:
                if max(width, height) < cfg.ROI.ROI_SIZE_THRS:
                    continue
            else:
                if max(width, height) >= cfg.ROI.ROI_SIZE_THRS:
                    continue
            real_indices.append(roi_index)

        real_num_rois = len(real_indices)
        if real_num_rois == 0:
            continue

        x_code_rois.append(pooled_feat[batch_index, real_indices])
        classes.append(fm_rois[batch_index, real_indices, 4].astype(int))
        bt_c_codes.append(raw_bt_c_codes[batch_index, real_indices])

    if len(classes) > 0:
        x_code_rois = torch.cat(x_code_rois, dim=0)
        classes = torch.from_numpy(np.concatenate(classes))
        bt_c_codes = torch.cat(bt_c_codes, dim=0)

    return x_code_rois, classes, bt_c_codes


def form_clabels_feat(clabels_emb, rois, num_rois):
    rois = rois.data.cpu().numpy()
    batch_size = rois.shape[0]
    num_rois = num_rois.data.cpu().numpy().tolist()
    max_num_roi = np.amax(num_rois)
    clabels_feat = Variable(torch.Tensor(batch_size, max_num_roi, 
        clabels_emb.size(1)).zero_())
    if cfg.CUDA:
        clabels_feat = clabels_feat.cuda()
    
    for i in range(batch_size):
        if num_rois[i] == 0:
            continue
        cat_ids = rois[i, :num_rois[i], 4]
        clabels_feat[i, :num_rois[i],:] = clabels_emb[cat_ids,:]

    # num_classes x max_num_roi x 50 (glove dim) 
    # -> num_classes x 50 (glove dim) x max_num_roi x 1
    clabels_feat = clabels_feat.transpose(1, 2).unsqueeze(3)
    
    return clabels_feat

def form_hmaps(raw_masks, num_rois, rois, hmap_size, num_classes):
    num_rois = num_rois.data.tolist()
    rois = rois.data.cpu().numpy()
    batch_size = int(raw_masks.size(0))

    re_raw_masks = F.interpolate(raw_masks, size=(hmap_size[-1], 
        hmap_size[-1]), mode='bilinear', align_corners=True)
    raw_gen_hmap = Variable(torch.zeros(batch_size, num_classes, 
        hmap_size[-1], hmap_size[-1]))
    if cfg.CUDA:
        raw_gen_hmap = raw_gen_hmap.cuda()

    for batch_index in range(batch_size):
        num_roi = num_rois[batch_index]
        cat_indices = rois[batch_index, :num_roi, 4].tolist()
        cat_indices = [int(cat_index) for cat_index in cat_indices]

        count = 0
        for cat_index in cat_indices:
            tmp_mask = re_raw_masks[batch_index, count]
            min_val = torch.min(tmp_mask)
            max_val = torch.max(tmp_mask)
            if min_val != max_val:
                tmp_mask = (tmp_mask-min_val)/(max_val-min_val)
            elif min_val == max_val and max_val < 0.6:
                tmp_mask = tmp_mask*0
            re_raw_masks[batch_index, count] = tmp_mask
            tmp_mask = tmp_mask.unsqueeze(0)
            orig_mask = raw_gen_hmap[batch_index, cat_index].unsqueeze(0)

            mask = torch.cat((tmp_mask, orig_mask), dim=0)
            mask = torch.max(mask, dim=0)[0]

            raw_gen_hmap[batch_index, cat_index] = mask
            count += 1

        for cat_index in cat_indices:
            mask = raw_gen_hmap[batch_index, cat_index]
            min_val = torch.min(mask)
            max_val = torch.max(mask)
            if min_val != max_val:
                mask = (mask-min_val)/(max_val-min_val)
            elif min_val == max_val and max_val < 0.6:
                mask = mask*0
            raw_gen_hmap[batch_index, cat_index] = mask

    gen_hmaps, gen_bt_masks = [], []
    for branch_index in range(cfg.TREE.BRANCH_NUM):
        tmp_raw_gen_hmap = F.interpolate(raw_gen_hmap, size=(hmap_size[branch_index], 
            hmap_size[branch_index]), mode='bilinear', align_corners=True)

        tmp_re_raw_masks = F.interpolate(re_raw_masks, size=(hmap_size[branch_index], 
            hmap_size[branch_index]), mode='bilinear', align_corners=True)
        
        gen_hmaps.append(tmp_raw_gen_hmap)
        gen_bt_masks.append(tmp_re_raw_masks)

    gen_fm_bt_masks = F.interpolate(re_raw_masks, size=(hmap_size[0]//2, 
            hmap_size[0]//2), mode='bilinear', align_corners=True)

    return gen_hmaps, gen_bt_masks, gen_fm_bt_masks

################################### FID score #################################
def get_activations(images, model, batch_size, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    #d0 = images.shape[0]
    d0 = int(images.size(0))
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, cfg.TEST.FID_DIMS))
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        '''batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        batch = Variable(batch, volatile=True)
        if cfg.CUDA:
            batch = batch.cuda()'''
        batch = images[start:end]

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_activation_statistics(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- act      : Numpy array of dimension (n_images, dim (e.g. 2048)).

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an 
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an 
               representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)