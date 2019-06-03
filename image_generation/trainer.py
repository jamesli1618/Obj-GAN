from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.utils import weights_init, load_params, copy_G_params
from miscc.utils import compute_inception_score, negative_log_posterior_probability
from miscc.utils import form_clabels_feat
from model import G_NET, PAT_D_NET64, PAT_D_NET128, PAT_D_NET256
from model import SHP_D_NET64, SHP_D_NET128, SHP_D_NET256, OBJ_SS_D_NET, OBJ_LS_D_NET
from trainDataset import prepare_data
from model import RNN_ENCODER, CNN_ENCODER, INCEPTION_V3
from miscc.losses import words_loss
from miscc.losses import patD_loss, shpD_loss, objD_loss, G_loss, KL_loss

import os
import time
import numpy as np
import sys
from PIL import Image

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, dataset):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.score_dir = os.path.join(output_dir, 'Score')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.score_dir)

        if len(cfg.GPU_IDS) == 1 and cfg.GPU_IDS[0] >= 0:
            torch.cuda.set_device(0)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.print_interval = cfg.TRAIN.PRINT_INTERVAL
        self.display_interval = cfg.TRAIN.DISPLAY_INTERVAL

        self.n_words = dataset.n_words
        self.ixtoword = dataset.ixtoword
        self.cats_index_dict = dataset.cats_index_dict
        self.cat_labels = dataset.cat_labels
        self.cat_label_lens = dataset.cat_label_lens
        self.sorted_cat_label_indices = dataset.sorted_cat_label_indices

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

        self.device = torch.device("cuda" if cfg.CUDA else "cpu")

        self.inception_model = INCEPTION_V3()
        self.glove_emb = dataset.glove_embed
        if cfg.CUDA:
            self.inception_model.cuda()
            self.glove_emb.cuda()
            if len(cfg.GPU_IDS) > 1:
                self.inception_model = nn.DataParallel(self.inception_model)
                self.inception_model.to(self.device)
                self.glove_emb = nn.DataParallel(self.glove_emb)
                self.glove_emb.to(self.device)
        self.inception_model.eval()
        self.glove_emb.eval()

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netG = G_NET(len(self.cats_index_dict))
        netsPatD, netsShpD = [], []
        if cfg.TREE.BRANCH_NUM > 0:
            netsPatD.append(PAT_D_NET64())
            netsShpD.append(SHP_D_NET64(len(self.cats_index_dict)))
        if cfg.TREE.BRANCH_NUM > 1:
            netsPatD.append(PAT_D_NET128())
            netsShpD.append(SHP_D_NET128(len(self.cats_index_dict)))
        if cfg.TREE.BRANCH_NUM > 2:
            netsPatD.append(PAT_D_NET256())
            netsShpD.append(SHP_D_NET256(len(self.cats_index_dict)))

        netObjSSD = OBJ_SS_D_NET(len(self.cats_index_dict))
        netObjLSD = OBJ_LS_D_NET(len(self.cats_index_dict))

        netG.apply(weights_init)
        netObjSSD.apply(weights_init)
        netObjLSD.apply(weights_init)
        for i in range(len(netsPatD)):
            netsPatD[i].apply(weights_init)
            netsShpD[i].apply(weights_init)
        print('# of netsPatD', len(netsPatD))
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            netObjSSD.cuda()
            netObjLSD.cuda()
            for i in range(len(netsPatD)):
                netsPatD[i].cuda()
                netsShpD[i].cuda()

            if len(cfg.GPU_IDS) > 1:
                text_encoder = nn.DataParallel(text_encoder)
                text_encoder.to(self.device)
                image_encoder = nn.DataParallel(image_encoder)
                image_encoder.to(self.device)
                netG = nn.DataParallel(netG)
                netG.to(self.device)
                netObjSSD = nn.DataParallel(netObjSSD)
                netObjSSD.to(self.device)
                netObjLSD = nn.DataParallel(netObjLSD)
                netObjLSD.to(self.device)
                for i in range(len(netsPatD)):
                    netsPatD[i] = nn.DataParallel(netsPatD[i])
                    netsPatD[i].to(self.device)

                    netsShpD[i] = nn.DataParallel(netsShpD[i])
                    netsShpD[i].to(self.device)
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1

            Gname = cfg.TRAIN.NET_G
            for i in range(len(netsPatD)):
                s_tmp = Gname[:Gname.rfind('/')]

                Dname = '%s/netPatD%d.pth' % (s_tmp, i)
                print('Load PatD from: ', Dname)
                state_dict = \
                    torch.load(Dname, map_location=lambda storage, loc: storage)
                netsPatD[i].load_state_dict(state_dict)

                Dname = '%s/netShpD%d.pth' % (s_tmp, i)
                print('Load ShpD from: ', Dname)
                state_dict = \
                    torch.load(Dname, map_location=lambda storage, loc: storage)
                netsShpD[i].load_state_dict(state_dict)

            s_tmp = Gname[:Gname.rfind('/')]
            Dname = '%s/netObjSSD.pth' % (s_tmp)
            print('Load ObjSSD from: ', Dname)
            state_dict = \
                torch.load(Dname, map_location=lambda storage, loc: storage)
            netObjSSD.load_state_dict(state_dict)

            s_tmp = Gname[:Gname.rfind('/')]
            Dname = '%s/netObjLSD.pth' % (s_tmp)
            print('Load ObjLSD from: ', Dname)
            state_dict = \
                torch.load(Dname, map_location=lambda storage, loc: storage)
            netObjLSD.load_state_dict(state_dict)

        return [text_encoder, image_encoder, netG, netsPatD, netsShpD, netObjSSD, netObjLSD, epoch]

    def define_optimizers(self, netG, netsPatD, netsShpD, netObjSSD, netObjLSD):
        optimizersPatD, optimizersShpD = [], []
        num_PatDs, num_ShpDs = len(netsPatD), len(netsShpD)
        for i in range(num_PatDs):
            opt = optim.Adam(netsPatD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersPatD.append(opt)

        for i in range(num_ShpDs):
            opt = optim.Adam(netsShpD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersShpD.append(opt)

        optimizerObjSSD = optim.Adam(netObjSSD.parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))

        optimizerObjLSD = optim.Adam(netObjLSD.parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersPatD, optimizersShpD, optimizerObjSSD, optimizerObjLSD

    def prepare_cat_emb(self):
        raw_clabels_emb = self.glove_emb(self.cat_labels.view(-1))
        raw_clabels_emb = raw_clabels_emb.detach().view(self.cat_labels.size(0), 
            self.cat_labels.size(1), -1)
        clabels_emb = Variable(torch.Tensor(raw_clabels_emb.size(0), 
            raw_clabels_emb.size(2)).zero_())
        if cfg.CUDA:
            clabels_emb = clabels_emb.cuda()
        for label_index in range(len(self.cats_index_dict)):
            label_len = int(self.cat_label_lens[label_index])
            if label_len == 1:
                clabels_emb[label_index] = raw_clabels_emb[label_index, 0]
            elif label_len == 2:
                clabels_emb[label_index] = (raw_clabels_emb[label_index, 0] + \
                    raw_clabels_emb[label_index, 1]) / 2.
        clabels_emb = clabels_emb[self.sorted_cat_label_indices]
        return clabels_emb

    def prepare_labels(self):
        match_labels = Variable(torch.LongTensor(range(self.batch_size)))
        if cfg.CUDA:
            match_labels = match_labels.cuda()

        return match_labels

    def save_model(self, netG, avg_param_G, netsPatD, netsShpD, netObjSSD, netObjLSD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsPatD)):
            netPatD = netsPatD[i]
            torch.save(netPatD.state_dict(),
                '%s/netPatD%d.pth' % (self.model_dir, i))

        for i in range(len(netsShpD)):
            netShpD = netsShpD[i]
            torch.save(netShpD.state_dict(),
                '%s/netShpD%d.pth' % (self.model_dir, i))

        torch.save(netObjSSD.state_dict(),
            '%s/netObjSSD.pth' % (self.model_dir))

        torch.save(netObjLSD.state_dict(),
            '%s/netObjLSD.pth' % (self.model_dir))
        print('Save G/Ds models.')

    def save_img_results(self, netG, noise, sent_emb, words_embs, glove_words_embs, 
        clabels_feat, mask, hmaps, rois, fm_rois, num_rois, bt_masks, fm_bt_masks, 
        image_encoder, captions, cap_lens, gen_iterations, name='current'):
        # Save images
        glb_max_num_roi = int(torch.max(num_rois))
        fake_imgs, _, attention_maps, bt_attention_maps, _, _ = netG(noise, 
            sent_emb, words_embs, glove_words_embs, clabels_feat, mask, 
            hmaps, rois, fm_rois, num_rois, bt_masks, fm_bt_masks, glb_max_num_roi)

        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None

            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

            bt_attn_maps = bt_attention_maps[i]
            att_sze = bt_attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   bt_attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/bt_G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps, _ = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self):
        text_encoder, image_encoder, netG, netsPatD, netsShpD, netObjSSD, netObjLSD, \
            start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)

        optimizerG, optimizersPatD, optimizersShpD, optimizerObjSSD, optimizerObjLSD = \
            self.define_optimizers(netG, netsPatD, netsShpD, netObjSSD, netObjLSD)

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        match_labels = self.prepare_labels()
        clabels_emb = self.prepare_cat_emb()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            predictions = []
            while step < self.num_batches:
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, glove_captions, cap_lens, hmaps, rois, fm_rois, \
                    num_rois, bt_masks, fm_bt_masks, class_ids, keys = prepare_data(data)

                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                max_len = int(torch.max(cap_lens))
                words_embs, sent_emb = text_encoder(captions, cap_lens, max_len)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                # glove_words_embs: batch_size x 50 (glove dim) x seq_len
                glove_words_embs = self.glove_emb(glove_captions.view(-1))
                glove_words_embs = glove_words_embs.detach().view(
                    glove_captions.size(0), glove_captions.size(1), -1)
                glove_words_embs = glove_words_embs[:,:num_words].transpose(1, 2)

                # clabels_feat: batch x 50 (glove dim) x max_num_roi x 1
                clabels_feat = form_clabels_feat(clabels_emb, rois[0], num_rois)

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                glb_max_num_roi = int(torch.max(num_rois))
                fake_imgs, bt_c_codes, _, _, mu, logvar = netG(noise, sent_emb, words_embs, 
                    glove_words_embs, clabels_feat, mask, hmaps, rois, fm_rois, num_rois, 
                    bt_masks, fm_bt_masks, glb_max_num_roi)
                bt_c_codes = [bt_c_code.detach() for bt_c_code in bt_c_codes]

                #######################################################
                # (3-1) Update PatD network
                ######################################################
                errPatD_total = 0
                PatD_logs = ''
                for i in range(len(netsPatD)):
                    netsPatD[i].zero_grad()
                    errPatD = patD_loss(netsPatD[i], imgs[i], fake_imgs[i], sent_emb)
                    errPatD.backward()
                    optimizersPatD[i].step()
                    errPatD_total += errPatD
                    PatD_logs += 'errPatD%d: %.2f ' % (i, errPatD.item())

                #######################################################
                # (3-2) Update ShpD network
                ######################################################
                errShpD_total = 0
                ShpD_logs = ''
                for i in range(len(netsShpD)):
                    netsShpD[i].zero_grad()
                    hmap = hmaps[i]
                    roi = rois[i]
                    errShpD = shpD_loss(netsShpD[i], imgs[i], fake_imgs[i], hmap, roi, num_rois)
                    errShpD.backward()
                    optimizersShpD[i].step()
                    errShpD_total += errShpD
                    ShpD_logs += 'errShpD%d: %.2f ' % (i, errShpD.item())

                #######################################################
                # (3-3) Update ObjSSD network
                ######################################################
                netObjSSD.zero_grad()
                errObjSSD = objD_loss(netObjSSD, imgs[-1], fake_imgs[-1], hmaps[-1], 
                	clabels_emb, bt_c_codes[-1], rois[0], num_rois)
                if float(errObjSSD) > 0:
                    errObjSSD.backward()
                    optimizerObjSSD.step()
                    ObjSSD_logs = 'errSSACD: %.2f ' % (errObjSSD.item())

                #######################################################
                # (3-4) Update ObjLSD network
                ######################################################
                netObjLSD.zero_grad()
                errObjLSD = objD_loss(netObjLSD, imgs[-1], fake_imgs[-1], hmaps[-1], 
                	clabels_emb, bt_c_codes[-1], fm_rois, num_rois, is_large_scale=True)
                if float(errObjLSD) > 0:
                    errObjLSD.backward()
                    optimizerObjLSD.step()
                    ObjLSD_logs = 'errObjLSD: %.2f ' % (errObjLSD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                netG.zero_grad()
                errG_total, G_logs = \
                    G_loss(netsPatD, netsShpD, netObjSSD, netObjLSD, image_encoder, fake_imgs, 
                                   hmaps, words_embs, sent_emb, clabels_emb, bt_c_codes[-1], 
                                   match_labels, cap_lens, class_ids, rois[0], fm_rois, num_rois)

                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                #######################################################
                # (5) Print and display
                ######################################################
                images = fake_imgs[-1].detach()
                pred = self.inception_model(images)
                predictions.append(pred.data.cpu().numpy())

                step += 1
                gen_iterations += 1

                if gen_iterations % self.print_interval == 0:
                    print('[%d/%d][%d]'%(epoch, self.max_epoch, gen_iterations) + '\n' + PatD_logs + 
                    	'\n' + ShpD_logs + '\n' + ObjSSD_logs + '\n' + ObjLSD_logs + '\n' + G_logs)
                # save images
                if gen_iterations % self.display_interval == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                        words_embs, glove_words_embs, clabels_feat, mask, 
                        hmaps, rois, fm_rois, num_rois, bt_masks, fm_bt_masks, 
                        image_encoder, captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)

            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_PatD: %.2f Loss_ShpD: %.2f Loss_ObjSSD: %.2f Loss_ObjLSD: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errPatD_total.item(), errShpD_total.item(), errObjSSD.item(), errObjLSD.item(), 
                     errG_total.item(), end_t - start_t))

            predictions = np.concatenate(predictions, 0)
            mean, std = compute_inception_score(predictions, min(10, self.batch_size))
            mean_conf, std_conf = \
                negative_log_posterior_probability(predictions, min(10, self.batch_size))

            fullpath = '%s/scores_%d.txt' % (self.score_dir, epoch)
            with open(fullpath, 'w') as fp:
                fp.write('mean, std, mean_conf, std_conf \n')
                fp.write('%f, %f, %f, %f' %(mean, std, mean_conf, std_conf))

            print('inception_score: mean, std, mean_conf, std_conf')
            print('inception_score: %f, %f, %f, %f' %(mean, std, mean_conf, std_conf))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsPatD, netsShpD, netObjSSD, netObjLSD, epoch)

        self.save_model(netG, avg_param_G, netsPatD, netsShpD, netObjSSD, netObjLSD, self.max_epoch)