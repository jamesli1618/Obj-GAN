from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from miscc.config import cfg
from miscc.load import load_filenames, load_text_data, load_sample_filenames
from miscc.load import load_glove_emb, load_cat_label, load_class_id, load_cats
from miscc.load import load_imgs_data, load_anns_data
from miscc.load import get_imgs, get_caption, get_hmaps_rois, get_gen_rois

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy.random as random
import os

class TrainDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
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

        self.glove_captions, self.glove_ixtoword, self.glove_wordtoix, \
            self.glove_embed = load_glove_emb(data_dir, split, train_names, test_names)

        self.cat_labels, self.cat_label_lens, self.sorted_cat_label_indices\
            = load_cat_label(data_dir, self.glove_wordtoix)

        self.class_id = load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)
        self.cats_dict, self.cats_index_dict = load_cats(data_dir, self.wordtoix)
        self.img_bytes = load_imgs_data(data_dir, split, self.filenames)
        self.insanns_dict = load_anns_data(data_dir, split, '_gt_insanns.pickle', 
                'gt', self.filenames, self.imsize, self.fmsize, self.cats_index_dict)

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        data_dir = self.data_dir
        #
        imgs = get_imgs(self.img_bytes[index], self.imsize, normalize=self.norm)

        hmaps, _, _, _, rois, fm_rois, num_rois, bt_masks, fm_bt_masks = get_hmaps_rois(
            self.insanns_dict[key], self.imsize, self.fmsize, self.cats_index_dict)

        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, glove_caps, cap_len = get_caption(self.captions, self.glove_captions, new_sent_ix)

        return imgs, caps, glove_caps, cap_len, hmaps, rois, fm_rois, num_rois, \
        bt_masks, fm_bt_masks, cls_id, key

    def __len__(self):
        return len(self.filenames)

def prepare_data(data):
    imgs, captions, glove_captions, captions_lens, hmaps, rois, fm_rois, \
        num_rois, bt_masks, fm_bt_masks, class_ids, keys = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)

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

    fm_rois = fm_rois[sorted_cap_indices]
    fm_bt_masks = fm_bt_masks[sorted_cap_indices]
    captions = captions[sorted_cap_indices].squeeze()
    glove_captions = glove_captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    keys = [keys[i] for i in sorted_cap_indices.numpy()]

    if cfg.CUDA:
        fm_rois = Variable(fm_rois).cuda()
        fm_bt_masks = Variable(fm_bt_masks.float()).cuda()
        captions = Variable(captions).cuda()
        glove_captions = Variable(glove_captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        num_rois = Variable(num_rois).cuda()
    else:
        fm_rois = Variable(fm_rois)
        fm_bt_masks = Variable(fm_bt_masks.float())
        captions = Variable(captions)
        glove_captions = Variable(glove_captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
        num_rois = Variable(num_rois)

    return [real_imgs, captions, glove_captions, sorted_cap_lens, real_hmaps, real_rois,
        fm_rois, num_rois, real_bt_masks, fm_bt_masks, class_ids, keys]