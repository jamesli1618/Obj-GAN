from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from miscc.utils import path_leaf
from model import G_NET, INS_D_NET, GLB_D_NET, vgg19_bn
from datasets import prepare_data

from miscc.losses import ins_discriminator_loss, glb_discriminator_loss, generator_loss
import os
import time
import numpy as np
import sys

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
        self.cats_dict = dataset.cats_dict
        self.cats_index_dict = dataset.cats_index_dict

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

        self.device = torch.device("cuda" if cfg.CUDA else "cpu")

        self.vgg_model = vgg19_bn(pretrained=True)
        if cfg.CUDA:
            self.vgg_model = self.vgg_model.cuda()
            if len(cfg.GPU_IDS) > 1:
                self.vgg_model = nn.DataParallel(self.vgg_model)
                self.vgg_model.to(self.device)
        self.vgg_model.eval()

    def build_models(self):
        netG = G_NET(len(self.cats_index_dict))
        netINSD = INS_D_NET(len(self.cats_index_dict))
        netGLBD = GLB_D_NET(len(self.cats_index_dict))

        netG.apply(weights_init)
        netINSD.apply(weights_init)
        netGLBD.apply(weights_init)
        
        if cfg.CUDA:
            netG.cuda()
            netINSD.cuda()
            netGLBD.cuda()

            if len(cfg.GPU_IDS) > 1:
                netG = nn.DataParallel(netG)
                netG.to(self.device)
                netINSD = nn.DataParallel(netINSD)
                netINSD.to(self.device)
                netGLBD = nn.DataParallel(netGLBD)
                netGLBD.to(self.device)

        # ########################################################### #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            filename = path_leaf(cfg.TRAIN.NET_G)
            istart = filename.rfind('_') + 1
            iend = filename.rfind('.')
            epoch = filename[istart:iend]
            epoch = int(epoch) + 1
            
            Gname = cfg.TRAIN.NET_G
            s_tmp = Gname[:Gname.rfind('/')]
            Dname = '%s/netINSD.pth' % (s_tmp)
            print('Load INSD from: ', Dname)
            state_dict = \
                torch.load(Dname, map_location=lambda storage, loc: storage)
            netINSD.load_state_dict(state_dict)

            s_tmp = Gname[:Gname.rfind('/')]
            Dname = '%s/netGLBD.pth' % (s_tmp)
            print('Load GLBD from: ', Dname)
            state_dict = \
                torch.load(Dname, map_location=lambda storage, loc: storage)
            netGLBD.load_state_dict(state_dict)

        return [netG, netINSD, netGLBD, epoch]

    def define_optimizers(self, netG, netINSD, netGLBD, lr_rate):
        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR*lr_rate,
                                betas=(0.5, 0.999))

        optimizerINSD = optim.Adam(netINSD.parameters(),
                                lr=cfg.TRAIN.DISCRIMINATOR_LR*lr_rate,
                                betas=(0.5, 0.999))

        optimizerGLBD = optim.Adam(netGLBD.parameters(),
                                lr=cfg.TRAIN.DISCRIMINATOR_LR*lr_rate,
                                betas=(0.5, 0.999))

        return optimizerG, optimizerINSD, optimizerGLBD

    def save_model(self, netG, avg_param_G, netINSD, netGLBD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        torch.save(netINSD.state_dict(),
            '%s/netINSD.pth' % (self.model_dir))
        #
        torch.save(netGLBD.state_dict(),
            '%s/netGLBD.pth' % (self.model_dir))

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, imgs, bbox_maps_fwd, bbox_maps_bwd,
        bbox_fmaps, hmaps, rois, num_rois, gen_iterations, name='current'):
        # Save images
        font_max = 20
        font_size = 12

        imgs = imgs.cpu()
        fake_hmaps = netG(noise, bbox_maps_fwd, 
            bbox_maps_bwd, bbox_fmaps)

        fake_hmaps = fake_hmaps.squeeze().detach().cpu()
        hmaps = hmaps.squeeze().cpu()

        # prepare captions
        batch_size = fake_hmaps.size(0)
        captions = Variable(torch.zeros(batch_size, cfg.ROI.BOXES_NUM)).cuda()
        for batch_index in range(self.batch_size):
            for roi_index in range(num_rois[batch_index]):
                rela_cat_id = int(rois[batch_index, roi_index, 4])
                captions[batch_index,roi_index] = self.cats_dict[rela_cat_id][0]

        att_sze = fake_hmaps.size(2)
        img_set, _ = build_super_images(imgs, captions, self.ixtoword, fake_hmaps, 
            att_sze, lr_imgs=None, font_max=font_max, font_size=font_size,
            max_word_num=cfg.ROI.BOXES_NUM)

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/G_%s_%d.png'% (self.image_dir, name, gen_iterations)
            im.save(fullpath)

        img_set, _ = build_super_images(imgs, captions, self.ixtoword, hmaps, 
            att_sze, lr_imgs=None, font_max=font_max, font_size=font_size,
            max_word_num=cfg.ROI.BOXES_NUM)

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'% (self.image_dir, name, gen_iterations)
            im.save(fullpath)

        #
        img_set, _ = build_super_images2(imgs, captions, self.ixtoword, fake_hmaps, 
            att_sze, lr_imgs=None, font_max=font_max, font_size=font_size,
            max_word_num=cfg.ROI.BOXES_NUM)

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/G2_%s_%d.png'% (self.image_dir, name, gen_iterations)
            im.save(fullpath)

        img_set, _ = build_super_images2(imgs, captions, self.ixtoword, hmaps, 
            att_sze, lr_imgs=None, font_max=font_max, font_size=font_size,
            max_word_num=cfg.ROI.BOXES_NUM)

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D2_%s_%d.png'% (self.image_dir, name, gen_iterations)
            im.save(fullpath)


    def train(self):
        netG, netINSD, netGLBD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)

        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, cfg.ROI.BOXES_NUM, len(self.cats_index_dict)*4))
        fixed_noise = Variable(torch.FloatTensor(batch_size, cfg.ROI.BOXES_NUM, len(self.cats_index_dict)*4).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        gen_iterations = 0
        lr_rate = 1
        pcp_score = 0.
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            if epoch > 50 and lr_rate > cfg.TRAIN.GENERATOR_LR/10.:
                lr_rate *= 0.98
            optimizerG, optimizerINSD, optimizerGLBD = self.define_optimizers(
                netG, netINSD, netGLBD, lr_rate)
            data_iter = iter(self.data_loader)
            step = 0
            
            while step < self.num_batches:
                #print('step: ', step)
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, pooled_hmaps, hmaps, bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps, \
                    rois, fm_rois, num_rois, class_ids, keys = prepare_data(data)

                #######################################################
                # (2) Generate fake images
                ######################################################
                max_num_roi = int(torch.max(num_rois))
                noise.data.normal_(0, 1)
                fake_hmaps = netG(noise[:,:max_num_roi], bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps)

                #######################################################
                # (3-1) Update INSD network
                ######################################################
                errINSD = 0
                netINSD.zero_grad()
                errINSD = ins_discriminator_loss(netINSD, hmaps, fake_hmaps, bbox_maps_fwd)
                errINSD.backward()
                optimizerINSD.step()
                INSD_logs = 'errINSD: %.2f ' % (errINSD.item())

                #######################################################
                # (3-2) Update GLBD network
                ######################################################
                errGLBD = 0
                netGLBD.zero_grad()
                errGLBD = glb_discriminator_loss(netGLBD, pooled_hmaps, fake_hmaps, bbox_maps_fwd)
                errGLBD.backward()
                optimizerGLBD.step()
                GLBD_logs = 'errGLBD: %.2f ' % (errGLBD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                netG.zero_grad()
                errG_total, G_logs, item_pcp_score = generator_loss(netINSD, netGLBD, 
                    self.vgg_model, hmaps, fake_hmaps, bbox_maps_fwd)
                pcp_score += item_pcp_score

                errG_total.backward()
                # `clip_grad_norm` helps prevent
                # the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(netG.parameters(), cfg.TRAIN.RNN_GRAD_CLIP)
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % self.print_interval == 0:
                    elapsed = time.time() - start_t
                    print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                              .format(epoch, step, self.num_batches,
                                      elapsed * 1000. / self.print_interval))
                    print(INSD_logs + '\n' + GLBD_logs + '\n' + G_logs)
                    start_t = time.time()

                # save images
                if gen_iterations % self.display_interval == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise[:,:max_num_roi], imgs,
                        bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps, hmaps,
                        rois, num_rois, gen_iterations, name='average')
                    load_params(netG, backup_para)

            pcp_score /= float(self.num_batches)
            print('pcp_score: ', pcp_score)
            fullpath = '%s/scores_%d.txt' % (self.score_dir, epoch)
            with open(fullpath, 'w') as fp:
                fp.write('pcp_score %f'%(pcp_score))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netINSD, netGLBD, epoch)

        self.save_model(netG, avg_param_G, netINSD, netGLBD, self.max_epoch)


    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            netG = G_NET(len(self.cats_index_dict))
            netG.apply(weights_init)
            netG.eval()

            if cfg.CUDA:
                netG.cuda()

            if len(cfg.GPU_IDS) > 1:
                netG = nn.DataParallel(netG)
                netG.to(self.device)

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, cfg.ROI.BOXES_NUM, len(self.cats_index_dict)*4))
            noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0

            for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if step % 100 == 0:
                        print('step: ', step)
                    # if step > 50:
                    #     break
                    imgs, pooled_hmaps, hmaps, bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps, \
                        rois, fm_rois, num_rois, class_ids, keys = prepare_data(data)
                    num_rois = num_rois.data.cpu().numpy()

                    cats_list = []
                    for batch_index in range(self.batch_size):
                        cats = []
                        for roi_index in range(num_rois[batch_index]):
                            rela_cat_id = int(rois[batch_index, roi_index, 4])
                            abs_cat_id = self.cats_dict[rela_cat_id][0]
                            cat = self.ixtoword[abs_cat_id].encode('ascii', 'ignore').decode('ascii')
                            cats.append(cat)
                        cats_list.append(cats)

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    max_num_roi = max(num_rois)
                    noise.data.normal_(0, 1)
                    fake_hmaps = netG(noise[:,:max_num_roi], bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps)
                    fake_hmaps = fake_hmaps.repeat(1, 1, 3, 1, 1)
                    for j in range(batch_size):
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = 0
                        # for k in range(len(fake_imgs)):
                        im = fake_hmaps[j][k].data.cpu().numpy()

                        minV = im.min()
                        maxV = im.max()
                        im = (im - minV) / (maxV - minV)
                        im *= 255                       
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)

                        cat = cats_list[j][k]
                        fullpath = '{0}_{1}.png'.format(s_tmp, cat)
                        im.save(fullpath)
