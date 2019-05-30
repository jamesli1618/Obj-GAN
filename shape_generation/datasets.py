from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg
from miscc.utils import calc_sort_size
from pycocotools.coco import COCO

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import io
import os
import struct
import sys
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
import skimage.io
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def prepare_data(data):
    # pooled_hmaps: batch x hmap_size x hmap_size
    # hmaps: batch x max_num_roi x hmap_size x hmap_size
    # bbox_maps_fwd: batch x max_num_roi x class_num x hmap_size x hmap_size
    # bbox_maps_bwd: batch x max_num_roi x class_num x hmap_size x hmap_size
    # bbox_fmaps: batch x max_num_roi x fmap_size x fmap_size
    # rois: batch x cfg.ROI.BOXES_NUM x 6
    # fm_rois: batch x cfg.ROI.BOXES_NUM x 6
    imgs, pooled_hmaps, hmaps, bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps,\
        rois, fm_rois, num_rois, class_ids, keys = data

    # shorten the data by max_num_roi
    max_num_roi = torch.max(num_rois)
    hmaps = hmaps[:,:max_num_roi].unsqueeze(2)
    bbox_maps_fwd = bbox_maps_fwd[:,:max_num_roi]
    bbox_maps_bwd = bbox_maps_bwd[:,:max_num_roi]
    bbox_fmaps = bbox_fmaps[:,:max_num_roi]
    pooled_hmaps = pooled_hmaps.unsqueeze(1)

    rois = rois.numpy()
    if cfg.CUDA:
        imgs = Variable(imgs.float()).cuda()
        pooled_hmaps = Variable(pooled_hmaps.float()).cuda()
        hmaps = Variable(hmaps.float()).cuda()
        bbox_maps_fwd = Variable(bbox_maps_fwd.float()).cuda()
        bbox_maps_bwd = Variable(bbox_maps_bwd.float()).cuda()
        bbox_fmaps = Variable(bbox_fmaps.float()).cuda()
        num_rois = Variable(num_rois).cuda()
    else:
        imgs = Variable(imgs.float())
        pooled_hmaps = Variable(pooled_hmaps.float())
        hmaps = Variable(hmaps.float())
        bbox_maps_fwd = Variable(bbox_maps_fwd.float())
        bbox_maps_bwd = Variable(bbox_maps_bwd.float())
        bbox_fmaps = Variable(bbox_fmaps.float())
        num_rois = Variable(num_rois)

    class_ids = class_ids.numpy()
    keys = list(keys)

    return [imgs, pooled_hmaps, hmaps, bbox_maps_fwd, bbox_maps_bwd, 
    bbox_fmaps, rois, fm_rois, num_rois, class_ids, keys]

def get_imgs(img_bytes, imsize, bbox=None, normalize=None):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    ret = normalize(transforms.Resize((imsize, imsize))(img))
    return ret

def get_hmaps_rois(anno_dict, hmap_size, fmap_size, cats_index_dict):
    rois = anno_dict['rois']
    fm_rois = anno_dict['fm_rois']
    raw_masks = anno_dict['masks']
    raw_pooled_hmaps = anno_dict['pooled masks']
    raw_bbox_maps = anno_dict['bbox maps']
    raw_bbox_fmaps = anno_dict['bbox fmaps']
    num_rois = anno_dict['num_rois']

    hmaps = np.zeros(shape=(cfg.ROI.BOXES_NUM, hmap_size, hmap_size))
    bbox_maps_fwd = np.zeros(shape=(cfg.ROI.BOXES_NUM, len(cats_index_dict), hmap_size, hmap_size))
    bbox_maps_bwd = np.zeros(shape=(cfg.ROI.BOXES_NUM, len(cats_index_dict), hmap_size, hmap_size))
    bbox_fmaps = np.zeros(shape=(cfg.ROI.BOXES_NUM, fmap_size, fmap_size))
    pooled_hmaps = np.zeros(shape=(hmap_size, hmap_size))

    if num_rois > 0:
        hmaps[:num_rois] = raw_masks.copy()
        pooled_hmaps = raw_pooled_hmaps.copy()
        for roi_index in range(num_rois):
            rela_cat_id = int(rois[roi_index, 4])
            bbox_maps_fwd[roi_index, rela_cat_id] = raw_bbox_maps[roi_index].copy()
        bbox_maps_bwd = bbox_maps_fwd[::-1].copy()
        bbox_fmaps[:num_rois] = raw_bbox_fmaps.copy()

    return pooled_hmaps, hmaps, bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps, rois, fm_rois, num_rois

class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', base_size=64):
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.imsize = base_size
        self.fmsize = cfg.ROI.FM_SIZE

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.train_names = self.load_filenames(data_dir, 'train')
        self.test_names = self.load_filenames(data_dir, 'test')

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)
        self.cats_dict, self.cats_index_dict = self.load_cats()

        self.img_bytes = self.load_imgs_data(data_dir, split)
        self.insanns_dict = self.load_anns_data(data_dir, split)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())

                    if len(tokens) == 0:
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def write_imgs(self, data_dir, filenames, filepath):
        with open(filepath, 'wb') as wfid:
            for img_index in range(len(filenames)):
                if img_index % 500 == 0:
                    print('%07d / %07d'%(img_index, len(filenames)))

                img_name = '%s/images/%s.jpg' % (data_dir, filenames[img_index])
                with open(img_name, 'rb') as img_fid:
                    img_bytes = img_fid.read()

                wfid.write(struct.pack('i', len(img_bytes)))
                wfid.write(img_bytes)

    def read_imgs(self, data_dir, filenames, filepath):
        img_bytes = []
        print('start loading bigfile (%0.02f GB) into memory' % (os.path.getsize(filepath)/1024/1024/1024))
        with open(filepath, 'rb') as fid:
            for img_index in range(len(filenames)):
                img_bytes_len = struct.unpack('i', fid.read(4))[0]
                img_bytes.append(fid.read(img_bytes_len))

        return img_bytes

    def load_imgs_data(self, data_dir, split):
        if split == 'train':
            filenames = self.train_names
        else:  # split=='test'
            filenames = self.test_names

        filepath = os.path.join(data_dir, '%s_imgs.bigfile'%(split))
        if not os.path.isfile(filepath):
            print('writing %s imgs'%(split))
            self.write_imgs(data_dir, filenames, filepath)

        img_bytes = self.read_imgs(data_dir, filenames, filepath)

        return img_bytes

    def load_insanns(self, data_dir, filenames, split):
        print('creating %s insanns'%(split))
        if split == 'train':
            split_name = 'train'
        else:  # split=='test'
            split_name = 'val'
        annFile = os.path.join(data_dir, 'insanns', 'instances_%s2014.json'%(split_name))
        coco = COCO(annFile)

        insanns_dict = {}
        
        for img_index in range(len(filenames)):
            if img_index % 500 == 0:
                print('%07d / %07d'%(img_index, len(filenames)))
            ### 1. initialize the ann containers
            anno_dict = {}
            rois = np.zeros(shape=(cfg.ROI.BOXES_NUM, cfg.ROI.BOXES_DIM))
            fm_rois = np.zeros(shape=(cfg.ROI.BOXES_NUM, cfg.ROI.BOXES_DIM))

            ### 2. fetch image id and the corresponding annotation ids
            img_id = coco.getImgId(filenames[img_index])

            crowd_annIds, noncrowd_annIds = coco.getAnnIds(imgIds=img_id)

            ### 3. early finish if no annotations found
            if len(noncrowd_annIds) == 0:
                anno_dict['rois'] = rois
                anno_dict['fm_rois'] = fm_rois
                anno_dict['masks'] = None
                anno_dict['pooled masks'] = None
                anno_dict['bbox maps'] = None
                anno_dict['bbox fmaps'] = None
                anno_dict['num_rois'] = 0
                insanns_dict[filenames[img_index]] = anno_dict
                continue

            ### 4. load annotations
            noncrowd_anns = coco.loadAnns(noncrowd_annIds)

            ### 5. filter small annotations
            img_name = '%s/images/%s.jpg' % (data_dir, filenames[img_index])
            I = skimage.io.imread(img_name)
            img_height, img_width = I.shape[0], I.shape[1]
            
            scales = np.zeros(shape=(2))
            fm_scales = np.zeros(shape=(2))
            scales[0] = self.imsize/float(img_height)
            scales[1] = self.imsize/float(img_width)
            fm_scales[0] = self.fmsize/float(img_height)
            fm_scales[1] = self.fmsize/float(img_width)

            num_rois = len(noncrowd_anns)
            raw_rois = np.zeros((num_rois, 6))
            kept_indices = []
            count = 0
            for roi_index in range(num_rois):
                roi = noncrowd_anns[roi_index]['bbox']
                bbox_width, bbox_height = roi[2:4]
                scaled_width = scales[1]*bbox_width
                scaled_height = scales[0]*bbox_height

                if scaled_width < cfg.ROI.ROI_MIN_SIZE and scaled_height < cfg.ROI.ROI_MIN_SIZE:
                    continue

                kept_indices.append(roi_index)

                raw_rois[count, :4] = roi
                raw_rois[count, 4] = self.cats_index_dict[noncrowd_anns[roi_index]['category_id']]
                count += 1

            num_rois = len(kept_indices)
            raw_rois = raw_rois[:num_rois]

            ### 6. early finish if no annotations left
            if num_rois == 0:
                anno_dict['rois'] = rois
                anno_dict['fm_rois'] = fm_rois
                anno_dict['masks'] = None
                anno_dict['pooled masks'] = None
                anno_dict['bbox maps'] = None
                anno_dict['bbox fmaps'] = None
                anno_dict['num_rois'] = 0
                insanns_dict[filenames[img_index]] = anno_dict
                continue

            ### 7. refine annotations according to the kept_indices
            if num_rois > cfg.ROI.BOXES_NUM:
                raw_rois, sorted_indices = calc_sort_size(raw_rois)
                raw_rois = raw_rois[:cfg.ROI.BOXES_NUM,:]
                kept_indices = sorted_indices[:cfg.ROI.BOXES_NUM]
                num_rois = cfg.ROI.BOXES_NUM

            refined_noncrowd_annIds, refined_noncrowd_anns = [], []
            for roi_index in kept_indices:
                refined_noncrowd_annIds.append(noncrowd_annIds[roi_index])
                refined_noncrowd_anns.append(noncrowd_anns[roi_index])

            ### 8. assemble the rois container
            rois[:num_rois, :] = raw_rois
            rois[:, [0, 2]] = rois[:, [0, 2]]*scales[1]
            rois[:, [1, 3]] = rois[:, [1, 3]]*scales[0]

            fm_rois[:num_rois, :] = raw_rois
            fm_rois[:, [0, 2]] = fm_rois[:, [0, 2]]*fm_scales[1]
            fm_rois[:, [1, 3]] = fm_rois[:, [1, 3]]*fm_scales[0]

            ### 9. construct seg masks, pooled masks
            raw_masks = np.zeros((num_rois, self.imsize, self.imsize))
            max_img_size = max(img_height, img_width)
            for roi_index in range(num_rois):
                poly = refined_noncrowd_anns[roi_index]['segmentation']
                new_poly = []
                for poly_item in poly:
                    new_poly_item = np.clip(np.array(poly_item), a_min=0, a_max=(max_img_size-1))
                    new_poly_item = new_poly_item.tolist()
                    new_poly.append(new_poly_item)

                mask = coco.segToMask(new_poly, img_height, img_width)
                mask = mask.astype(float)
                re_mask = resize(mask, [self.imsize, self.imsize])
                raw_masks[roi_index] = re_mask
            raw_pooled_masks = np.amax(raw_masks, axis=0)

            ### 10. construct bbox maps
            raw_bbox_masks = np.zeros((num_rois, self.imsize, self.imsize))
            for roi_index in range(num_rois):
                x_start = min(int(round(rois[roi_index, 0])), self.imsize-1)
                y_start = min(int(round(rois[roi_index, 1])), self.imsize-1)
                x_end = min(int(round(rois[roi_index, 0]+rois[roi_index, 2])), self.imsize-1)
                y_end = min(int(round(rois[roi_index, 1]+rois[roi_index, 3])), self.imsize-1)
                raw_bbox_masks[roi_index, y_start:y_end, x_start:x_end] = 1

            ### 11. construct bbox maps
            raw_bbox_fmasks = np.zeros((num_rois, self.fmsize, self.fmsize))
            for roi_index in range(num_rois):
                x_start = min(int(round(fm_rois[roi_index, 0])), self.fmsize-1)
                y_start = min(int(round(fm_rois[roi_index, 1])), self.fmsize-1)
                x_end = min(int(round(fm_rois[roi_index, 0]+fm_rois[roi_index, 2])), self.fmsize-1)
                y_end = min(int(round(fm_rois[roi_index, 1]+fm_rois[roi_index, 3])), self.fmsize-1)
                raw_bbox_fmasks[roi_index, y_start:y_end, x_start:x_end] = 1

            anno_dict['rois'] = rois
            anno_dict['fm_rois'] = fm_rois
            anno_dict['masks'] = raw_masks
            anno_dict['pooled masks'] = raw_pooled_masks
            anno_dict['bbox maps'] = raw_bbox_masks
            anno_dict['bbox fmaps'] = raw_bbox_fmasks
            anno_dict['num_rois'] = num_rois
            insanns_dict[filenames[img_index]] = anno_dict

        return insanns_dict

    def load_anns_data(self, data_dir, split):
        if split == 'train':
            filenames = self.train_names
        else:  # split=='test'
            filenames = self.test_names

        filepath = os.path.join(data_dir, '%s_shape_insanns.pickle'%(split))
        if not os.path.isfile(filepath):
            insanns_dict = self.load_insanns(data_dir, filenames, split)

            with open(filepath, 'wb') as f:
                pickle.dump([insanns_dict], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                insanns_dict = x[0]
                del x
                print('Load from: ', filepath)

        return insanns_dict

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, self.train_names)
            test_captions = self.load_captions(data_dir, self.test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = self.train_names
        else:  # split=='test'
            captions = test_captions
            filenames = self.test_names

        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def load_cats(self):
        cats_path = os.path.join(self.data_dir, 'categories.txt')
        raw_cats = pd.read_csv(cats_path, header=None)
        cats_id = list(raw_cats.iloc[: , 0])
        cats_name = list(raw_cats.iloc[: , 1])
        cats_dict = {}
        cats_index_dict = {}
        tokenizer = RegexpTokenizer(r'\w+')
        for i in range(len(cats_id)):
            tokens = tokenizer.tokenize(cats_name[i].lower())
            tokens_new = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                t_emb = self.wordtoix[t]
                if len(t) > 0:
                    tokens_new.append(t_emb)

            cats_dict[i] = tokens_new
            cats_index_dict[cats_id[i]] = i
        return cats_dict, cats_index_dict

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir

        imgs = get_imgs(self.img_bytes[index], self.imsize, bbox, normalize=self.norm)

        pooled_hmaps, hmaps, bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps, \
            rois, fm_rois, num_rois = get_hmaps_rois(self.insanns_dict[key], 
            self.imsize, self.fmsize, self.cats_index_dict)

        return imgs, pooled_hmaps, hmaps, bbox_maps_fwd, \
            bbox_maps_bwd, bbox_fmaps, rois, fm_rois, num_rois, cls_id, key

    def __len__(self):
        return len(self.filenames)
