import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from miscc.config import cfg
from miscc.utils import permute_seg, feat_select

from GlobalAttention import func_attention


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids,
    batch_size, eps=1e-8, top1=True, is_training=True):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA and is_training:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
        if top1:
            _, predicted0 = torch.max(scores0, 1)
            _, predicted1 = torch.max(scores1, 1)
            correct = ((predicted0 == labels).sum().cpu().item() +
                       (predicted1 == labels).sum().cpu().item())
            accuracy = (100. * correct) / (batch_size * 2.)
        else:
            # s0(i, j): the similarity between the i-th image and the j-th text
            # s1(i, j): the similarity between the i-th text and the j-th image
            # accuracy = [scores0, scores1]
            # accuracy = [nn.Softmax()(scores0), nn.Softmax()(scores1)]
            accuracy = scores0
        # print('s_correct = ', correct, 's_accuracy = ', accuracy)
    else:
        loss0, loss1, accuracy = None
    return loss0, loss1, accuracy


def words_loss(img_features, words_emb, labels, cap_lens, 
    class_ids, batch_size, top1=True, is_training=True):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA and is_training:
            masks = masks.cuda()

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        if top1:
            _, predicted = torch.max(similarities, 1)
            _, predicted1 = torch.max(similarities1, 1)
            correct = ((predicted == labels).sum().cpu().item() +
                       (predicted1 == labels).sum().cpu().item())
            accuracy = (100. * correct) / (batch_size * 2.)
            # print('w_correct = ', correct, 'w_accuracy = ', accuracy)
        else:
            # similarities(i, j): the similarity between the i-th image and the j-th text
            # similarities1(i, j): the similarity between the i-th text and the j-th image
            # accuracy = [similarities, similarities1]
            accuracy = [nn.Softmax()(similarities), nn.Softmax()(similarities1)]
    else:
        loss0, loss1, accuracy = None, None, None
    return loss0, loss1, att_maps, accuracy


# ##################Loss for G and Ds##############################
def patD_loss(netPatD, real_imgs, fake_imgs, conditions):
    # Forward
    real_features = netPatD(real_imgs)
    fake_features = netPatD(fake_imgs.detach())
    # loss
    #
    if len(cfg.GPU_IDS) > 1:
        cond_real_logits = netPatD.module.COND_DNET(real_features, conditions)
        cond_fake_logits = netPatD.module.COND_DNET(fake_features, conditions)
    else:
        cond_real_logits = netPatD.COND_DNET(real_features, conditions)
        cond_fake_logits = netPatD.COND_DNET(fake_features, conditions)

    real_labels = Variable(torch.FloatTensor(cond_real_logits.size()).fill_(1))
    fake_labels = Variable(torch.FloatTensor(cond_fake_logits.size()).fill_(0))
    if cfg.CUDA:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()

    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    if len(cfg.GPU_IDS) > 1:
        cond_wrong_logits = netPatD.module.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    else:
        cond_wrong_logits = netPatD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    uncond_flag = False
    if len(cfg.GPU_IDS) > 1:
        uncond_flag = netPatD.module.UNCOND_DNET is not None
    else:
        uncond_flag = netPatD.UNCOND_DNET is not None

    if uncond_flag:
        if len(cfg.GPU_IDS) > 1:
            real_logits = netPatD.module.UNCOND_DNET(real_features)
            fake_logits = netPatD.module.UNCOND_DNET(fake_features)
        else:
            real_logits = netPatD.UNCOND_DNET(real_features)
            fake_logits = netPatD.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = ((real_errD * cfg.TRAIN.SMOOTH.UNCOND_LAMBDA + cond_real_errD * cfg.TRAIN.SMOOTH.TXT_LAMBDA) / 2. +
                (fake_errD * cfg.TRAIN.SMOOTH.UNCOND_LAMBDA + (cond_fake_errD + cond_wrong_errD) * cfg.TRAIN.SMOOTH.TXT_LAMBDA) / 3.)
    else:
        errD = (cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.) * cfg.TRAIN.SMOOTH.TXT_LAMBDA
    return errD

def shpD_loss(netShpD, real_imgs, fake_imgs, seg_conditions, rois, num_rois):
    # Forward
    real_features = netShpD(real_imgs, seg_conditions)
    fake_features = netShpD(fake_imgs.detach(), seg_conditions)
    fake_seg_conditions, valid_mask = permute_seg(seg_conditions, rois, num_rois)
    # loss
    #
    if len(valid_mask) > 0:
        fake_features2 = netShpD(real_imgs[valid_mask], fake_seg_conditions[valid_mask])

    if len(cfg.GPU_IDS) > 1:
        real_logits = netShpD.module.UNCOND_DNET(real_features)
        fake_logits = netShpD.module.UNCOND_DNET(fake_features)
        if len(valid_mask) > 0:
            wrong_logits = netShpD.module.UNCOND_DNET(fake_features2)
    else:
        real_logits = netShpD.UNCOND_DNET(real_features)
        fake_logits = netShpD.UNCOND_DNET(fake_features)
        if len(valid_mask) > 0:
            wrong_logits = netShpD.UNCOND_DNET(fake_features2)
    #
    real_labels = Variable(torch.FloatTensor(real_logits.size()).fill_(1))
    fake_labels = Variable(torch.FloatTensor(fake_logits.size()).fill_(0))
    if cfg.CUDA:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()

    real_errD = nn.BCELoss()(real_logits, real_labels)
    fake_errD = nn.BCELoss()(fake_logits, fake_labels)
    if len(valid_mask) > 0:
        wrong_errD = nn.BCELoss()(wrong_logits, fake_labels[valid_mask])

    errD = real_errD
    if len(valid_mask) > 0:
        errD += ( fake_errD + wrong_errD ) / 2.
    else:
        errD += fake_errD

    return errD


def objD_loss(netObjD, real_imgs, fake_imgs, seg_conditions, 
    raw_conditions, raw_bt_c_codes, fm_rois, num_rois, is_large_scale=False):
    real_pooled_feat = netObjD(real_imgs, seg_conditions, fm_rois, num_rois)    
    real_features, classes, bt_c_codes = feat_select(real_pooled_feat, raw_bt_c_codes, 
        fm_rois, num_rois, is_large_scale=is_large_scale)
    
    fake_pooled_feat = netObjD(fake_imgs.detach(), seg_conditions, fm_rois, num_rois)
    fake_features, _, _ = feat_select(fake_pooled_feat, raw_bt_c_codes, 
        fm_rois, num_rois, is_large_scale=is_large_scale)

    fake_seg_conditions, valid_mask = permute_seg(seg_conditions, fm_rois, num_rois)
    classes2 = []
    if len(valid_mask) > 0:
        fake_pooled_feat2 = netObjD(real_imgs[valid_mask], fake_seg_conditions[valid_mask], 
            fm_rois[valid_mask], num_rois[valid_mask])
        fake_features2, classes2, bt_c_codes2 = feat_select(fake_pooled_feat2, raw_bt_c_codes, 
            fm_rois[valid_mask], num_rois[valid_mask], is_large_scale=is_large_scale)

    errD = 0
    real_batch_size = len(classes)
    if real_batch_size == 0:
        return errD

    conditions = Variable(torch.FloatTensor(real_batch_size, raw_conditions.size(1)))
    if cfg.CUDA:
        conditions = conditions.cuda()
    for class_index in range(real_batch_size):
        conditions[class_index] = raw_conditions[classes[class_index]]

    conditions = torch.cat((conditions, bt_c_codes), dim=1)

    if len(cfg.GPU_IDS) > 1:
        cond_real_logits = netObjD.module.COND_DNET(real_features, conditions)
        cond_fake_logits = netObjD.module.COND_DNET(fake_features, conditions)
    else:
        cond_real_logits = netObjD.COND_DNET(real_features, conditions)
        cond_fake_logits = netObjD.COND_DNET(fake_features, conditions)

    real_labels = Variable(torch.FloatTensor(cond_real_logits.size()).fill_(1))
    fake_labels = Variable(torch.FloatTensor(cond_fake_logits.size()).fill_(0))
    if cfg.CUDA:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()

    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)

    if real_batch_size > 1:
        if len(cfg.GPU_IDS) > 1:
            cond_wrong_logits = netObjD.module.COND_DNET(real_features[:(real_batch_size - 1)], conditions[1:real_batch_size])
        else:
            cond_wrong_logits = netObjD.COND_DNET(real_features[:(real_batch_size - 1)], conditions[1:real_batch_size])
        cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:real_batch_size])

    if len(valid_mask) > 0 and len(classes2) > 0:
        conditions2 = Variable(torch.FloatTensor(len(classes2), raw_conditions.size(1)))
        if cfg.CUDA:
            conditions2 = conditions2.cuda()
        for class_index in range(len(classes2)):
            conditions2[class_index] = raw_conditions[classes[class_index]]

        conditions2 = torch.cat((conditions2, bt_c_codes2), dim=1)

        if len(cfg.GPU_IDS) > 1:
            cond_wrong_logits2 = netObjD.module.COND_DNET(fake_features2, conditions2)
        else:
            cond_wrong_logits2 = netObjD.COND_DNET(fake_features2, conditions2)
        fake_labels2 = Variable(torch.FloatTensor(cond_wrong_logits2.size()).fill_(0))
        if cfg.CUDA:
            fake_labels2 = fake_labels2.cuda()
        cond_wrong_errD2 = nn.BCELoss()(cond_wrong_logits2, fake_labels2)

    uncond_flag = False
    if len(cfg.GPU_IDS) > 1:
        uncond_flag = netObjD.module.UNCOND_DNET is not None
    else:
        uncond_flag = netObjD.UNCOND_DNET is not None
    if uncond_flag:
        if len(cfg.GPU_IDS) > 1:
            real_logits = netObjD.module.UNCOND_DNET(real_features)
            fake_logits = netObjD.module.UNCOND_DNET(fake_features)
        else:
            real_logits = netObjD.UNCOND_DNET(real_features)
            fake_logits = netObjD.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD += (real_errD + cond_real_errD) / 2.

        tmp_errD = fake_errD + cond_fake_errD
        denorm = 3.
        if real_batch_size > 1:
            tmp_errD += cond_wrong_errD
        if len(valid_mask) > 0 and len(classes2) > 0:
            tmp_errD += cond_wrong_errD2
            denorm += 1.
        errD += tmp_errD / denorm
    else:
        errD += cond_real_errD
        tmp_errD = cond_fake_errD
        denorm = 2.
        if real_batch_size > 1:
            tmp_errD += cond_wrong_errD
        if len(valid_mask) > 0 and len(classes2) > 0:
            tmp_errD += cond_wrong_errD2
            denorm += 1.
        errD += tmp_errD / denorm

    return errD


def G_loss(netsPatD, netsShpD, netObjSSD, netObjLSD, image_encoder, fake_imgs, seg_conditions,
        words_embs, sent_emb, slabels_emb, raw_bt_c_codes, match_labels, cap_lens, class_ids, 
        rois, fm_rois, num_rois):
    numDs = len(netsPatD)
    batch_size = fake_imgs[0].size(0)
    logs = ''
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsPatD[i](fake_imgs[i])
        if len(cfg.GPU_IDS) > 1:
            cond_logits = netsPatD[i].module.COND_DNET(features, sent_emb)
        else:
            cond_logits = netsPatD[i].COND_DNET(features, sent_emb)

        real_labels = Variable(torch.FloatTensor(cond_logits.size()).fill_(1))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
        cond_errG = nn.BCELoss()(cond_logits, real_labels)

        uncond_flag = False
        if len(cfg.GPU_IDS) > 1:
            uncond_flag = netsPatD[i].module.UNCOND_DNET is not None
        else:
            uncond_flag = netsPatD[i].UNCOND_DNET is not None
        if uncond_flag:
            if len(cfg.GPU_IDS) > 1:
                logits = netsPatD[i].module.UNCOND_DNET(features)
            else:
                logits = netsPatD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            pat_g_loss = errG * cfg.TRAIN.SMOOTH.UNCOND_LAMBDA + cond_errG * cfg.TRAIN.SMOOTH.TXT_LAMBDA
        else:
            pat_g_loss = cond_errG
        errG_total += pat_g_loss
        logs += 'pat_g_loss %d: %.2f ' % (i, pat_g_loss.item())

        #
        features = netsShpD[i](fake_imgs[i], seg_conditions[i])

        if len(cfg.GPU_IDS) > 1:
            logits = netsShpD[i].module.UNCOND_DNET(features)
        else:
            logits = netsShpD[i].UNCOND_DNET(features)
        real_labels = Variable(torch.FloatTensor(logits.size()).fill_(1))
        if cfg.CUDA:
            real_labels = real_labels.cuda()

        errG = nn.BCELoss()(logits, real_labels)
        shp_g_loss = errG * cfg.TRAIN.SMOOTH.SHP_LAMBDA
        errG_total += shp_g_loss
        logs += 'shp_g_loss%d: %.2f ' % (i, shp_g_loss.item())

        # Ranking loss
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            region_features, cnn_code = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, _, _ = words_loss(region_features, words_embs,
                                             match_labels, cap_lens,
                                             class_ids, batch_size)
            w_loss = (w_loss0 + w_loss1) * \
                cfg.TRAIN.SMOOTH.DAMSM_LAMBDA
            # err_words = err_words + w_loss.data[0]

            s_loss0, s_loss1, _ = sent_loss(cnn_code, sent_emb,
                                         match_labels, class_ids, batch_size)
            s_loss = (s_loss0 + s_loss1) * \
                cfg.TRAIN.SMOOTH.DAMSM_LAMBDA
            # err_sent = err_sent + s_loss.data[0]

            errG_total += w_loss + s_loss
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss.item(), s_loss.item())

    objss_g_loss = 0
    fake_pooled_feat = netObjSSD(fake_imgs[-1], seg_conditions[-1], rois, num_rois)
    fake_features, classes, bt_c_codes = feat_select(fake_pooled_feat, raw_bt_c_codes, 
        rois, num_rois, is_large_scale=False)
    real_batch_size = len(classes)
    if real_batch_size > 0:
        conditions = Variable(torch.FloatTensor(real_batch_size, slabels_emb.size(1)))
        if cfg.CUDA:
            conditions = conditions.cuda()
        for class_index in range(real_batch_size):
            conditions[class_index] = slabels_emb[classes[class_index]]

        conditions = torch.cat((conditions, bt_c_codes), dim=1)

        if len(cfg.GPU_IDS) > 1:
            cond_logits = netObjSSD.module.COND_DNET(fake_features, conditions)
        else:
            cond_logits = netObjSSD.COND_DNET(fake_features, conditions)

        real_labels = Variable(torch.FloatTensor(cond_logits.size()).fill_(1))
        if cfg.CUDA:
            real_labels = real_labels.cuda()

        cond_errG = nn.BCELoss()(cond_logits, real_labels)

        uncond_flag = False
        if len(cfg.GPU_IDS) > 1:
            uncond_flag = netObjSSD.module.UNCOND_DNET is not None
        else:
            uncond_flag = netObjSSD.UNCOND_DNET is not None
        if uncond_flag:
            if len(cfg.GPU_IDS) > 1:
                logits = netObjSSD.module.UNCOND_DNET(fake_features)
            else:
                logits = netObjSSD.UNCOND_DNET(fake_features)
            errG = nn.BCELoss()(logits, real_labels)

            objss_g_loss += (cond_errG + errG) * cfg.TRAIN.SMOOTH.OBJ_LAMBDA
        else:
            objss_g_loss += cond_errG * cfg.TRAIN.SMOOTH.OBJ_LAMBDA


    objls_g_loss = 0
    fake_pooled_feat = netObjLSD(fake_imgs[-1], seg_conditions[-1], fm_rois, num_rois)
    fake_features, classes, bt_c_codes = feat_select(fake_pooled_feat, raw_bt_c_codes, 
        fm_rois, num_rois, is_large_scale=True)
    real_batch_size = len(classes)
    if real_batch_size > 0:
        conditions = Variable(torch.FloatTensor(real_batch_size, slabels_emb.size(1)))
        if cfg.CUDA:
            conditions = conditions.cuda()
        for class_index in range(real_batch_size):
            conditions[class_index] = slabels_emb[classes[class_index]]

        conditions = torch.cat((conditions, bt_c_codes), dim=1)

        if len(cfg.GPU_IDS) > 1:
            cond_logits = netObjLSD.module.COND_DNET(fake_features, conditions)
        else:
            cond_logits = netObjLSD.COND_DNET(fake_features, conditions)

        real_labels = Variable(torch.FloatTensor(cond_logits.size()).fill_(1))
        if cfg.CUDA:
            real_labels = real_labels.cuda()

        cond_errG = nn.BCELoss()(cond_logits, real_labels)

        uncond_flag = False
        if len(cfg.GPU_IDS) > 1:
            uncond_flag = netObjLSD.module.UNCOND_DNET is not None
        else:
            uncond_flag = netObjLSD.UNCOND_DNET is not None
        if uncond_flag:
            if len(cfg.GPU_IDS) > 1:
                logits = netObjLSD.module.UNCOND_DNET(fake_features)
            else:
                logits = netObjLSD.UNCOND_DNET(fake_features)
            errG = nn.BCELoss()(logits, real_labels)

            objls_g_loss += (cond_errG + errG) * cfg.TRAIN.SMOOTH.OBJ_LAMBDA
        else:
            objls_g_loss += cond_errG * cfg.TRAIN.SMOOTH.OBJ_LAMBDA

    if float(objss_g_loss) > 0:
        logs += 'objss_g_loss: %.2f ' % (objss_g_loss.item())
        errG_total += objss_g_loss

    if float(objls_g_loss) > 0:
        logs += 'objls_g_loss: %.2f ' % (objls_g_loss.item())
        errG_total += objls_g_loss

    return errG_total, logs


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
