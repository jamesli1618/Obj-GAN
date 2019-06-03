from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from miscc.config import cfg
from miscc.load import load_filenames, load_text_data, load_sample_filenames
from miscc.load import load_glove_emb, load_cat_label, load_class_id, load_cats
from miscc.load import load_imgs_data, load_acts_data, load_anns_data
from miscc.load import get_imgs, get_caption, get_hmaps_rois, get_gen_rois

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy.random as random
import os

class TestDataset(data.Dataset):
    def __init__(self, data_dir, split='test',
                 base_size=64):
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.fmsize = cfg.ROI.FM_SIZE 

        self.data = []
        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split)

        train_names = load_filenames(data_dir, 'train')
        test_names = load_filenames(data_dir, 'test')
        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words \
            = load_text_data(data_dir, split, train_names, test_names)

        if cfg.TEST.SAMPLE_VAL:
            self.filenames, self.sentids = load_sample_filenames(data_dir)

        self.glove_captions, self.glove_ixtoword, self.glove_wordtoix, \
            self.glove_embed = load_glove_emb(data_dir, split, train_names, test_names)

        self.cat_labels, self.cat_label_lens, self.sorted_cat_label_indices\
            = load_cat_label(data_dir, self.glove_wordtoix)

        self.class_id = load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)
        self.cats_dict, self.cats_index_dict = load_cats(data_dir, self.wordtoix)
        self.img_bytes = load_imgs_data(data_dir, split, self.filenames)
        self.acts_dict = load_acts_data(data_dir, split)

        if cfg.TEST.USE_GT_BOX_SEG <= 1:
            self.insanns_gt_dict = load_anns_data(data_dir, split, '_gt_insanns.pickle', 
                'gt', self.filenames, self.imsize, self.fmsize, self.cats_index_dict)
        elif cfg.TEST.USE_GT_BOX_SEG == 2: # use gen box and gen shape
            self.insanns_gen_dict = load_anns_data(data_dir, split, '_gen_insanns.pickle', 
                'gen', self.filenames, self.imsize, self.fmsize, self.cats_index_dict)
        else:
            print('Error: unrecognizable USE_GT_BOX_SEG option!')

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        data_dir = self.data_dir
        #
        imgs = get_imgs(self.img_bytes[index], self.imsize, normalize=self.norm)

        if self.acts_dict is None:
            return imgs, key

        acts = self.acts_dict[key]
        # random select a sentence
        if cfg.TEST.SAMPLE_VAL:
            new_sent_ix = self.sentids[index]
            sent_ix = new_sent_ix % self.embeddings_num
        else:
            sent_ix = random.randint(0, self.embeddings_num)
            new_sent_ix = index * self.embeddings_num + sent_ix
        caps, glove_caps, cap_len = get_caption(self.captions, self.glove_captions, new_sent_ix)
        #
        if cfg.TEST.USE_GT_BOX_SEG <= 1:
            hmaps, bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps, rois, fm_rois, num_rois, \
            bt_masks, fm_bt_masks = get_hmaps_rois(self.insanns_gt_dict[key], self.imsize, 
                self.fmsize, self.cats_index_dict)

            return imgs, acts, caps, glove_caps, cap_len, hmaps, bbox_maps_fwd, bbox_maps_bwd, \
                bbox_fmaps, rois, fm_rois, num_rois, bt_masks, fm_bt_masks, cls_id, key, new_sent_ix

        elif cfg.TEST.USE_GT_BOX_SEG == 2: # use gen box and gen seg
            bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps, rois, fm_rois, num_rois \
                = get_gen_rois(self.insanns_gen_dict[key], self.imsize, self.fmsize, 
                    self.cats_index_dict, sent_ix)

            return imgs, acts, caps, glove_caps, cap_len, bbox_maps_fwd, bbox_maps_bwd, \
                bbox_fmaps, rois, fm_rois, num_rois, cls_id, key, new_sent_ix


    def __len__(self):
        return len(self.filenames)

def prepare_data(data):
    # bbox_maps_fwd: batch x max_num_roi x class_num x hmap_size x hmap_size
    # bbox_maps_bwd: batch x max_num_roi x class_num x hmap_size x hmap_size
    # bbox_fmaps: batch x max_num_roi x fmap_size x fmap_size
    # rois[0]: batch x cfg.ROI.BOXES_NUM x 6
    # fm_rois: batch x cfg.ROI.BOXES_NUM x 6

    imgs, acts, captions, glove_captions, captions_lens, hmaps, bbox_maps_fwd, \
        bbox_maps_bwd, bbox_fmaps, rois, fm_rois, num_rois, bt_masks, fm_bt_masks, \
        class_ids, keys, sent_ids = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    num_rois = num_rois[sorted_cap_indices]
    real_hmaps, real_imgs, real_bt_masks, real_rois = [], [], [], []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        rois[i] = rois[i][sorted_cap_indices]
        bt_masks[i] = bt_masks[i][sorted_cap_indices]
        hmaps[i] = hmaps[i][sorted_cap_indices]

        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
            real_rois.append(Variable(rois[i]).cuda())
            real_bt_masks.append(Variable(bt_masks[i].float()).cuda())
            real_hmaps.append(Variable(hmaps[i].float()).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))
            real_rois.append(Variable(rois[i]))
            real_bt_masks.append(Variable(bt_masks[i].float()))
            real_hmaps.append(Variable(hmaps[i].float()))

    acts = acts[sorted_cap_indices].numpy()
    captions = captions[sorted_cap_indices].squeeze()
    glove_captions = glove_captions[sorted_cap_indices].squeeze()
    # shorten the data by max_num_roi
    max_num_roi = torch.max(num_rois)
    bbox_maps_fwd = bbox_maps_fwd[sorted_cap_indices,:max_num_roi]
    bbox_maps_bwd = bbox_maps_bwd[sorted_cap_indices,:max_num_roi]
    bbox_fmaps = bbox_fmaps[sorted_cap_indices,:max_num_roi]
    fm_rois = fm_rois[sorted_cap_indices]
    fm_bt_masks = fm_bt_masks[sorted_cap_indices]
    class_ids = class_ids[sorted_cap_indices].numpy()
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    sent_ids = [sent_ids[i] for i in sorted_cap_indices.numpy()]

    if cfg.CUDA:
        captions = Variable(captions).cuda()
        glove_captions = Variable(glove_captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        num_rois = Variable(num_rois).cuda()
        bbox_maps_fwd = Variable(bbox_maps_fwd.float()).cuda()
        bbox_maps_bwd = Variable(bbox_maps_bwd.float()).cuda()
        bbox_fmaps = Variable(bbox_fmaps.float()).cuda()
        fm_rois = Variable(fm_rois).cuda()
        fm_bt_masks = Variable(fm_bt_masks.float()).cuda()
    else:
        captions = Variable(captions)
        glove_captions = Variable(glove_captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
        num_rois = Variable(num_rois)
        bbox_maps_fwd = Variable(bbox_maps_fwd.float())
        bbox_maps_bwd = Variable(bbox_maps_bwd.float())
        bbox_fmaps = Variable(bbox_fmaps.float())
        fm_rois = Variable(fm_rois)
        fm_bt_masks = Variable(fm_bt_masks.float())

    return [real_imgs, acts, captions, glove_captions, sorted_cap_lens, real_hmaps, 
        bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps, real_rois, fm_rois, num_rois, 
        real_bt_masks, fm_bt_masks, class_ids, keys, sent_ids]


def prepare_gen_data(data):
    # bbox_maps_fwd: batch x max_num_roi x class_num x hmap_size x hmap_size
    # bbox_maps_bwd: batch x max_num_roi x class_num x hmap_size x hmap_size
    # bbox_fmaps: batch x max_num_roi x fmap_size x fmap_size
    # rois[0]: batch x cfg.ROI.BOXES_NUM x 6
    # fm_rois: batch x cfg.ROI.BOXES_NUM x 6

    imgs, acts, captions, glove_captions, captions_lens, bbox_maps_fwd, bbox_maps_bwd, \
        bbox_fmaps, rois, fm_rois, num_rois, class_ids, keys, sent_ids = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    num_rois = num_rois[sorted_cap_indices]
    real_imgs, real_rois = [], []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        rois[i] = rois[i][sorted_cap_indices]

        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
            real_rois.append(Variable(rois[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))
            real_rois.append(Variable(rois[i]))

    acts = acts[sorted_cap_indices].numpy()
    captions = captions[sorted_cap_indices].squeeze()
    glove_captions = glove_captions[sorted_cap_indices].squeeze()
    # shorten the data by max_num_roi
    max_num_roi = torch.max(num_rois)
    bbox_maps_fwd = bbox_maps_fwd[sorted_cap_indices,:max_num_roi]
    bbox_maps_bwd = bbox_maps_bwd[sorted_cap_indices,:max_num_roi]
    bbox_fmaps = bbox_fmaps[sorted_cap_indices,:max_num_roi]
    fm_rois = fm_rois[sorted_cap_indices]
    class_ids = class_ids[sorted_cap_indices].numpy()
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    sent_ids = [sent_ids[i] for i in sorted_cap_indices.numpy()]

    if cfg.CUDA:
        captions = Variable(captions).cuda()
        glove_captions = Variable(glove_captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        num_rois = Variable(num_rois).cuda()
        bbox_maps_fwd = Variable(bbox_maps_fwd.float()).cuda()
        bbox_maps_bwd = Variable(bbox_maps_bwd.float()).cuda()
        bbox_fmaps = Variable(bbox_fmaps.float()).cuda()
        fm_rois = Variable(fm_rois).cuda()
    else:
        captions = Variable(captions)
        glove_captions = Variable(glove_captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
        num_rois = Variable(num_rois)
        bbox_maps_fwd = Variable(bbox_maps_fwd.float())
        bbox_maps_bwd = Variable(bbox_maps_bwd.float())
        bbox_fmaps = Variable(bbox_fmaps.float())
        fm_rois = Variable(fm_rois)

    return [real_imgs, acts, captions, glove_captions, sorted_cap_lens, 
        bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps, real_rois, fm_rois, 
        num_rois, class_ids, keys, sent_ids]


def prepare_acts_data(data):
    imgs, keys = data

    real_imgs = []
    for i in range(len(imgs)):
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    return [real_imgs, keys]