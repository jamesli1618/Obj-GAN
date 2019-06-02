from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from miscc.config import cfg
from miscc.utils import mkdir_p, denorm_imgs, weights_init
from miscc.utils import build_super_images, build_super_shape_images
from miscc.utils import compute_inception_score, negative_log_posterior_probability
from miscc.utils import form_clabels_feat, form_hmaps
from miscc.utils import get_activations, calculate_activation_statistics, calculate_frechet_distance
from miscc.losses import words_loss, sent_loss
from model import G_NET, SHP_G_NET, RNN_ENCODER, CNN_ENCODER, INCEPTION_V3, INCEPTION_V3_FID
from testDataset import prepare_data, prepare_gen_data, prepare_acts_data

import os
import time
import numpy as np
import sys
from PIL import Image
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

# ################# Text to image task############################ #
class condGANEvaluator(object):
    def __init__(self, output_dir, data_loader, dataset):
        #if cfg.TRAIN.FLAG:
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        self.snapshot_dir = os.path.join(output_dir, 'Snapshot')
        self.score_dir = os.path.join(output_dir, 'Score')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)
        mkdir_p(self.snapshot_dir)
        mkdir_p(self.score_dir)

        if len(cfg.GPU_IDS) == 1 and cfg.GPU_IDS[0] >= 0:
            torch.cuda.set_device(0)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.display_interval = cfg.TRAIN.DISPLAY_INTERVAL
        self.device = torch.device("cuda" if cfg.CUDA else "cpu")

        self.n_words = dataset.n_words
        self.ixtoword = dataset.ixtoword
        self.cats_dict = dataset.cats_dict
        self.cats_index_dict = dataset.cats_index_dict
        self.cat_labels = dataset.cat_labels
        self.cat_label_lens = dataset.cat_label_lens
        self.sorted_cat_label_indices = dataset.sorted_cat_label_indices

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

        self.glove_emb = dataset.glove_embed
        if cfg.CUDA:
            self.glove_emb.cuda()
            if len(cfg.GPU_IDS) > 1:
                self.glove_emb = nn.DataParallel(self.glove_emb)
                self.glove_emb.to(self.device)
        self.glove_emb.eval()

        if cfg.TEST.USE_TF:
            import miscc.inception_score_tf as inception_score
            self.inception_score = inception_score
            torch.cuda.set_device(0)
        else:
            self.inception_model = INCEPTION_V3()
            block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[cfg.TEST.FID_DIMS]
            self.inception_model_fid = INCEPTION_V3_FID([block_idx])
            if cfg.CUDA:
                self.inception_model.cuda()
                self.inception_model_fid.cuda()
                if len(cfg.GPU_IDS) > 1:
                    self.inception_model = nn.DataParallel(self.inception_model)
                    self.inception_model.to(self.device)
                    self.inception_model_fid = nn.DataParallel(self.inception_model_fid)
                    self.inception_model_fid.to(self.device)
            self.inception_model.eval()
            self.inception_model_fid.eval()

    def build_models(self):
        # ############################## encoders ############################# #
        text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        # ########### image generator and (potential) shape generator ########## #
        netG = G_NET(len(self.cats_index_dict))
        netG.apply(weights_init)
        netG.eval()
        netShpG = None
        if cfg.TEST.USE_GT_BOX_SEG > 0:
            netShpG = SHP_G_NET(len(self.cats_index_dict))
            netShpG.apply(weights_init)
            netShpG.eval()

        # ################### parallization and initialization ################## #
        if cfg.CUDA:
            text_encoder.cuda()
            image_encoder.cuda()
            netG.cuda()
            if cfg.TEST.USE_GT_BOX_SEG > 0:
                netShpG.cuda()

            if len(cfg.GPU_IDS) > 1:
                text_encoder = nn.DataParallel(text_encoder)
                text_encoder.to(self.device)
                image_encoder = nn.DataParallel(image_encoder)
                image_encoder.to(self.device)
                netG = nn.DataParallel(netG)
                netG.to(self.device)

            if cfg.TEST.USE_GT_BOX_SEG > 0:
                netShpG = nn.DataParallel(netShpG)
                netShpG.to(self.device)

        state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        print('Load G from: ', cfg.TRAIN.NET_G)
        
        if cfg.TEST.USE_GT_BOX_SEG > 0:
            state_dict = torch.load(cfg.TEST.NET_SHP_G, map_location=lambda storage, loc: storage)
            netShpG.load_state_dict(state_dict)
            print('Load Shape G from: ', cfg.TEST.NET_SHP_G)

        return [text_encoder, image_encoder, netG, netShpG]

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
        match_labels = Variable(torch.LongTensor(range(cfg.TEST.RP_POOL_SIZE)))
        return match_labels

    def save_img_results(self, fake_imgs, attn_maps, bt_attn_maps, 
        captions, cap_lens, gen_iterations):
        font_max = [50, 50]
        font_size = [30, 50]
        batch_size = fake_imgs[0].size(0)
        # Save images
        for i in range(len(attn_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None

            attn_maps = attn_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword, attn_maps, att_sze, 
                    lr_imgs=lr_img, font_max=font_max[i], font_size=font_size[i], batch_size=batch_size)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%d_%d.png'\
                    % (self.snapshot_dir, gen_iterations, i)
                im.save(fullpath)


            bt_attn_maps = bt_attn_maps[i]
            att_sze = bt_attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword, bt_attn_maps, att_sze, 
                    lr_imgs=lr_img, font_max=font_max[i], font_size=font_size[i], batch_size=batch_size)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/bt_G_%d_%d.png'\
                    % (self.snapshot_dir, gen_iterations, i)
                im.save(fullpath)

    def save_shape_results(self, imgs, hmaps, rois, num_rois, gen_iterations, model_type):
        # model_type: ['D', 'G']
        # Save images
        font_max = 20
        font_size = 12
        imgs = imgs.cpu()
        hmaps = hmaps.cpu()
        num_rois = num_rois.data.cpu().tolist()
         # prepare captions
        batch_size = hmaps.size(0)
        captions = Variable(torch.zeros(batch_size, cfg.ROI.BOXES_NUM)).cuda()
        for batch_index in range(self.batch_size):
            for roi_index in range(num_rois[batch_index]):
                rela_cat_id = int(rois[batch_index, roi_index, 4])
                captions[batch_index,roi_index] = self.cats_dict[rela_cat_id][0]
        att_sze = hmaps.size(2)
        img_set, _ = build_super_shape_images(imgs, captions, self.ixtoword, hmaps, 
            att_sze, lr_imgs=None, font_max=font_max, font_size=font_size,
            max_word_num=cfg.ROI.BOXES_NUM, batch_size=batch_size)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/Shape%s_%d.png'% (self.snapshot_dir, model_type, gen_iterations)
            im.save(fullpath)

    def save_singleimages(self, images, keys, sent_ids):
        images = images.detach()
        for i in range(images.size(0)):
            fullpath = '%s/%s_%d.jpg' % (self.image_dir, keys[i], sent_ids[i])
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def dump_fid_acts(self, data_dir, split):
        filepath = os.path.join(data_dir, '%s_acts_tf%d.pickle'%(split, cfg.TEST.USE_TF))
        if os.path.isfile(filepath):
            return

        acts_dict = {}
        count = 0
        for step, data in enumerate(self.data_loader, 0):
            if count % 10 == 0:
                print('%07d / %07d'%(count, self.num_batches))
            imgs, keys = prepare_acts_data(data)
            batch_size = len(keys)
            if cfg.TEST.USE_TF:
                denorm_images = denorm_imgs(imgs[-1])
                acts = self.inception_score.get_fid_pred(denorm_images)
            else:
                acts = get_activations(imgs[-1], self.inception_model_fid, batch_size)

            for batch_index in range(batch_size):
                acts_dict[keys[batch_index]] = acts[batch_index]

            count += 1

        with open(filepath, 'wb') as f:
            pickle.dump([acts_dict], f, protocol=2)
            print('Save to: ', filepath)

    def evaluate(self, split_dir, hmap_size):
        text_encoder, image_encoder, netG, netShpG = self.build_models()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise_img = Variable(torch.FloatTensor(batch_size, nz))
        if cfg.CUDA:
            noise_img = noise_img.cuda()

        if cfg.TEST.USE_GT_BOX_SEG > 0:
            noise_shp = Variable(torch.FloatTensor(batch_size, cfg.ROI.BOXES_NUM, 
                len(self.cats_index_dict)*4))
            if cfg.CUDA:
                noise_shp = noise_shp.cuda()

        match_labels = self.prepare_labels()
        clabels_emb = self.prepare_cat_emb()

        predictions, fake_acts_set, acts_set, w_accuracy, s_accuracy = [], [], [], [], []
        region_features_set, cnn_code_set, words_embs_set, sent_emb_set, \
            class_ids_set, cap_lens_set = [], [], [], [], [], []
        gen_iterations = 0
        rp_count = 0

        for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
            for step, data in enumerate(self.data_loader, 0):
                #######################################################
                # (1) Prepare general test data
                #######################################################
                if cfg.TEST.USE_GT_BOX_SEG < 2:
                    imgs, acts, captions, glove_captions, cap_lens, gt_hmaps, bbox_maps_fwd, \
                        bbox_maps_bwd, bbox_fmaps, rois, fm_rois, num_rois, gt_bt_masks, \
                        gt_fm_bt_masks, class_ids, keys, sent_ids = prepare_data(data)
                else:
                    imgs, acts, captions, glove_captions, cap_lens, bbox_maps_fwd, \
                        bbox_maps_bwd, bbox_fmaps, rois, fm_rois, num_rois, \
                        class_ids, keys, sent_ids = prepare_gen_data(data)
                    gt_hmaps = None

                #######################################################
                # (2) Prepare real shapes or generate fake shapes
                #######################################################
                batch_size = len(num_rois)
                max_num_roi = int(torch.max(num_rois))
                noise_img = noise_img[:batch_size].data.normal_(0, 1)

                if cfg.TEST.USE_GT_BOX_SEG > 0: # 1 for gt box and gen shape, 2 for gen box and gen shape
                    noise_shp = noise_shp[:batch_size].data.normal_(0, 1)
                    raw_masks = netShpG(noise_shp[:,:max_num_roi], bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps)
                    raw_masks = raw_masks.squeeze(2).detach()
                    if gen_iterations % self.display_interval == 0:
                        self.save_shape_results(imgs[0], raw_masks, rois[0], 
                            num_rois, gen_iterations, model_type='G')
                        if gt_hmaps is not None:
                            self.save_shape_results(imgs[0], gt_hmaps[0].squeeze(), rois[0], 
                                num_rois, gen_iterations, model_type='D')
                    gen_hmaps, gen_bt_masks, gen_fm_bt_masks = form_hmaps(raw_masks, num_rois, rois[0], 
                        hmap_size, len(self.cats_index_dict))
                    hmaps = gen_hmaps
                    bt_masks = gen_bt_masks
                    fm_bt_masks = gen_fm_bt_masks
                else: # 0 for gt box and gt shape
                    hmaps = gt_hmaps
                    bt_masks = gt_bt_masks
                    fm_bt_masks = gt_fm_bt_masks

                #######################################################
                # (3) Prepare or compute text embeddings
                #######################################################
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
                # (4) Generate fake images
                #######################################################
                fake_imgs, _, attn_maps, bt_attn_maps, mu, logvar = netG(noise_img, 
                    sent_emb, words_embs, glove_words_embs, clabels_feat, mask, 
                    hmaps, rois, fm_rois, num_rois, bt_masks, fm_bt_masks, max_num_roi)

                if gen_iterations % self.display_interval == 0:
                    if cfg.TEST.SAVE_OPTIONS == 'SNAPSHOT':
                        self.save_img_results(fake_imgs, attn_maps, bt_attn_maps, 
                            captions, cap_lens, gen_iterations)
                    elif cfg.TEST.SAVE_OPTIONS == 'IMAGE':
                        self.save_singleimages(fake_imgs[-1], keys, sent_ids)
                    print('%d / %d'%(gen_iterations, self.num_batches))

                #######################################################
                # (5) Prepare intermediate results for evaluation
                #######################################################
                images = fake_imgs[-1].detach()

                region_features, cnn_code = image_encoder(images)
                region_features, cnn_code = region_features.detach(), cnn_code.detach()

                if rp_count >= cfg.TEST.RP_POOL_SIZE:
                    region_features_set = torch.cat(region_features_set, dim=0)
                    region_features_set = region_features_set[:cfg.TEST.RP_POOL_SIZE]
                    cnn_code_set = torch.cat(cnn_code_set, dim=0)
                    cnn_code_set = cnn_code_set[:cfg.TEST.RP_POOL_SIZE]

                    sent_emb_set = torch.cat(sent_emb_set, dim=0)
                    sent_emb_set = sent_emb_set[:cfg.TEST.RP_POOL_SIZE]
                    class_ids_set = np.concatenate(class_ids_set, 0)
                    class_ids_set = class_ids_set[:cfg.TEST.RP_POOL_SIZE]
                    cap_lens_set = torch.cat(cap_lens_set, dim=0)
                    cap_lens_set = cap_lens_set[:cfg.TEST.RP_POOL_SIZE]

                    max_len = int(torch.max(cap_lens_set))
                    new_words_embs_set = torch.zeros(rp_count, sent_emb_set.size(1), max_len)
                    accum = 0
                    for tmp_words_embs in words_embs_set:
                        tmp_bs, tmp_max_len = tmp_words_embs.size(0), tmp_words_embs.size(2)
                        new_words_embs_set[accum:accum+tmp_bs, :, :tmp_max_len] = tmp_words_embs
                        accum += tmp_bs
                    new_words_embs_set = new_words_embs_set[:cfg.TEST.RP_POOL_SIZE]

                    _, _, _, w_accu = words_loss(region_features_set, new_words_embs_set,
                        match_labels, cap_lens_set, class_ids_set, cfg.TEST.RP_POOL_SIZE,
                        is_training=False)
                    _, _, s_accu = sent_loss(cnn_code_set, sent_emb_set, match_labels, 
                        class_ids_set, cfg.TEST.RP_POOL_SIZE, is_training=False)
                    w_accuracy.append(w_accu)
                    s_accuracy.append(s_accu)

                    rp_count = 0
                    region_features_set, cnn_code_set, words_embs_set, sent_emb_set, \
                        class_ids_set, cap_lens_set = [], [], [], [], [], []
                else:
                    region_features_set.append(region_features.cpu())
                    cnn_code_set.append(cnn_code.cpu())
                    words_embs_set.append(words_embs.cpu())
                    sent_emb_set.append(sent_emb.cpu())
                    class_ids_set.append(class_ids)
                    cap_lens_set.append(cap_lens.cpu())
                    rp_count += batch_size

                if cfg.TEST.USE_TF:
                    denorm_images = denorm_imgs(images)
                    pred = self.inception_score.get_inception_pred(denorm_images)
                else:
                    pred = self.inception_model(images)
                    pred = pred.data.cpu().numpy()
                predictions.append(pred)

                if cfg.TEST.USE_TF:
                    fake_acts = self.inception_score.get_fid_pred(denorm_images)
                else:
                    fake_acts = get_activations(images, self.inception_model_fid, batch_size)
                acts_set.append(acts)
                fake_acts_set.append(fake_acts)

                gen_iterations += 1
                if gen_iterations >= cfg.TEST.TEST_IMG_NUM:
                    break

        if cfg.TEST.USE_TF:
            self.inception_score.close_sess()

        #######################################################
        # (6) Evaluation
        #######################################################

        predictions = np.concatenate(predictions, 0)
        mean, std = compute_inception_score(predictions, min(10, self.batch_size))
        mean_conf, std_conf = \
            negative_log_posterior_probability(predictions, min(10, self.batch_size))
        accu_w, std_w, accu_s, std_s = np.mean(w_accuracy), np.std(w_accuracy), np.mean(s_accuracy), np.std(s_accuracy)

        acts_set = np.concatenate(acts_set, 0)
        fake_acts_set = np.concatenate(fake_acts_set, 0)
        real_mu, real_sigma = calculate_activation_statistics(acts_set)
        fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)
        fid_score = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)

        fullpath = '%s/scores.txt' % (self.score_dir)
        with open(fullpath, 'w') as fp:
            fp.write('mean, std, mean_conf, std_conf, accu_w, std_w, accu_s, std_s, fid_score \n')
            fp.write('%f, %f, %f, %f, %f, %f, %f, %f, %f' %(mean, std, 
                mean_conf, std_conf, accu_w, std_w, accu_s, std_s, fid_score))

        print('inception_score: mean, std, mean_conf, std_conf, accu_w, std_w, accu_s, std_s, fid_score')
        print('inception_score: %f, %f, %f, %f, %f, %f, %f, %f, %f' %(mean, std, 
            mean_conf, std_conf, accu_w, std_w, accu_s, std_s, fid_score))