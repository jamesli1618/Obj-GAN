from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg
from miscc.utils import path_leaf, calc_sort_size, is_non_zero_file
from pycocotools.coco import COCO

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchtext

import io
import os
import struct
import sys
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from skimage.transform import resize
import skimage.io
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import spacy
spacy_en = spacy.load('en')

################################### auxiliary functions #################################
def build_dictionary(train_captions, test_captions):
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
        # rev.append(0)  # do not need '<end>' token
        train_captions_new.append(rev)

    test_captions_new = []
    for t in test_captions:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        test_captions_new.append(rev)

    return [train_captions_new, test_captions_new, ixtoword, wordtoix, len(ixtoword)]


def write_imgs(data_dir, filenames, filepath):
    with open(filepath, 'wb') as wfid:
        for img_index in range(len(filenames)):
            if img_index % 500 == 0:
                print('%07d / %07d'%(img_index, len(filenames)))

            img_name = '%s/images/%s.jpg' % (data_dir, filenames[img_index])
            with open(img_name, 'rb') as img_fid:
                img_bytes = img_fid.read()

            wfid.write(struct.pack('i', len(img_bytes)))
            wfid.write(img_bytes)


def read_imgs(data_dir, filenames, filepath):
    img_bytes = []
    print('start loading bigfile (%0.02f GB) into memory' % (os.path.getsize(filepath)/1024/1024/1024))
    with open(filepath, 'rb') as fid:
        for img_index in range(len(filenames)):
            img_bytes_len = struct.unpack('i', fid.read(4))[0]
            img_bytes.append(fid.read(img_bytes_len))

    return img_bytes

def translate_glove(captions, wordtoix):
    captions_new = []
    for t in captions:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        captions_new.append(rev)
    return captions_new

def en_tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


def get_caption(captions, glove_captions, sent_ix):
    # a list of indices for a sentence
    sent_caption = np.asarray(captions[sent_ix]).astype('int64')
    sent_glove_caption = np.asarray(glove_captions[sent_ix]).astype('int64')
    if len(sent_caption) > len(sent_glove_caption):
        sent_caption = sent_caption[:len(sent_glove_caption)]
    else:
        sent_glove_caption = sent_glove_caption[:len(sent_caption)]
        
    if (sent_caption == 0).sum() > 0:
        print('ERROR: do not need END (0) token', sent_caption)
    num_words = len(sent_caption)
    # pad with 0s (i.e., '<end>')
    x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
    glove_x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
    x_len = num_words
    if num_words <= cfg.TEXT.WORDS_NUM:
        x[:num_words, 0] = sent_caption
        glove_x[:num_words, 0] = sent_glove_caption
    else:
        ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
        np.random.shuffle(ix)
        ix = ix[:cfg.TEXT.WORDS_NUM]
        ix = np.sort(ix)
        x[:, 0] = sent_caption[ix]
        glove_x[:, 0] = sent_glove_caption[ix]
        x_len = cfg.TEXT.WORDS_NUM
    return x, glove_x, x_len

def get_imgs(img_bytes, imsize, normalize=None):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    width, height = img.size

    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        re_img = transforms.Resize((imsize[i], imsize[i]))(img)
        ret.append(normalize(re_img))

    return ret

def get_hmaps_rois(anno_dict, hmap_size, fmap_size, cats_index_dict):
    rois = anno_dict['rois']
    fm_rois = anno_dict['fm_rois']
    raw_masks = anno_dict['masks']
    raw_bbox_maps = anno_dict['bbox maps']
    raw_bbox_fmaps = anno_dict['bbox fmaps']
    num_rois = anno_dict['num_rois']

    hmaps, bt_masks = [], []
    for branch_index in range(cfg.TREE.BRANCH_NUM):
        hmaps.append(np.zeros(shape=(len(cats_index_dict), hmap_size[branch_index], hmap_size[branch_index])))
        bt_masks.append(np.zeros(shape=(cfg.ROI.BOXES_NUM, hmap_size[branch_index], hmap_size[branch_index])))
    fm_bt_masks = np.zeros(shape=(cfg.ROI.BOXES_NUM, hmap_size[0]//2, hmap_size[0]//2))

    for roi_index in range(num_rois):
        mask = raw_masks[roi_index]
        rela_cat_id = int(rois[0][roi_index, 4])

        re_mask = resize(mask, [hmap_size[0]//2, hmap_size[0]//2])
        fm_bt_masks[roi_index, :, :] = re_mask

        for branch_index in range(cfg.TREE.BRANCH_NUM):
            re_mask = resize(mask, [hmap_size[branch_index], hmap_size[branch_index]])
            bt_masks[branch_index][roi_index, :, :] = re_mask
            hmaps[branch_index][rela_cat_id, :, :] += re_mask
 
    bbox_maps_fwd = np.zeros(shape=(cfg.ROI.BOXES_NUM, len(cats_index_dict), 
        hmap_size[0], hmap_size[0]))
    bbox_maps_bwd = np.zeros(shape=(cfg.ROI.BOXES_NUM, len(cats_index_dict), 
        hmap_size[0], hmap_size[0]))
    bbox_fmaps = np.zeros(shape=(cfg.ROI.BOXES_NUM, fmap_size, fmap_size))

    if num_rois > 0:
        for roi_index in range(num_rois):
            rela_cat_id = int(rois[0][roi_index, 4])
            bbox_maps_fwd[roi_index, rela_cat_id] = raw_bbox_maps[roi_index].copy()
        bbox_maps_bwd = bbox_maps_fwd[::-1].copy()
        bbox_fmaps[:num_rois] = raw_bbox_fmaps.copy()

    return hmaps, bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps, rois, fm_rois, num_rois, bt_masks, fm_bt_masks


def get_gen_rois(anno_dicts, hmap_size, fmap_size, cats_index_dict, sent_ix):
    rois = anno_dicts[sent_ix]['rois']
    fm_rois = anno_dicts[sent_ix]['fm_rois']
    raw_bbox_maps = anno_dicts[sent_ix]['bbox maps']
    raw_bbox_fmaps = anno_dicts[sent_ix]['bbox fmaps']
    num_rois = anno_dicts[sent_ix]['num_rois']
 
    bbox_maps_fwd = np.zeros(shape=(cfg.ROI.BOXES_NUM, len(cats_index_dict), 
        hmap_size[0], hmap_size[0]))
    bbox_maps_bwd = np.zeros(shape=(cfg.ROI.BOXES_NUM, len(cats_index_dict), 
        hmap_size[0], hmap_size[0]))
    bbox_fmaps = np.zeros(shape=(cfg.ROI.BOXES_NUM, fmap_size, fmap_size))

    if num_rois > 0 and raw_bbox_maps is not None:
        for roi_index in range(num_rois):
            rela_cat_id = int(rois[0][roi_index, 4])
            bbox_maps_fwd[roi_index, rela_cat_id] = raw_bbox_maps[roi_index].copy()
        bbox_maps_bwd = bbox_maps_fwd[::-1].copy()
        bbox_fmaps[:num_rois] = raw_bbox_fmaps.copy()

    return bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps, rois, fm_rois, num_rois

################################### load text data #################################
def load_text_data(data_dir, split, train_names, test_names):
    filepath = os.path.join(data_dir, 'captions.pickle')
    if not os.path.isfile(filepath):
        train_captions = load_captions(data_dir, train_names)
        test_captions = load_captions(data_dir, test_names)

        train_captions, test_captions, ixtoword, wordtoix, n_words = \
            build_dictionary(train_captions, test_captions)
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
        filenames = train_names
    else:  # split=='test'
        captions = test_captions
        filenames = test_names
    return filenames, captions, ixtoword, wordtoix, n_words


def load_captions(data_dir, filenames, embeddings_num):
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
                # print('tokens', tokens)
                if len(tokens) == 0:
                    print('cap', cap)
                    continue

                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                all_captions.append(tokens_new)
                cnt += 1
                if cnt == embeddings_num:
                    break
            if cnt < embeddings_num:
                print('ERROR: the captions for %s less than %d'
                      % (filenames[i], cnt))
    return all_captions


################################### load cat data #################################
def load_cats(data_dir, wordtoix):
    cats_path = '%s/categories.txt' % (data_dir)
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
            t_emb = wordtoix[t]
            if len(t) > 0:
                tokens_new.append(t_emb)
        #assert len(tokens_new) <= attn_cfg.ROI.BOX_WORDS_NUM
        cats_dict[i] = tokens_new
        cats_index_dict[cats_id[i]] = i
    return cats_dict, cats_index_dict

def load_cat_label(data_dir, glove_wordtoix):
    cat_labels, cat_label_lens = [], []
    cat_label_path = '%s/categories.txt' % (data_dir)
    with open(cat_label_path, "r") as f:
        raw_cats = f.read().split('\n')
        for raw_cat in raw_cats:
            if len(raw_cat) == 0:
                continue
            raw_cat = raw_cat.replace("\ufffd\ufffd", " ")
            # picks out sequences of alphanumeric characters as tokens
            # and drops everything else
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(raw_cat.lower())
            tokens = tokens[1:]
            # print('tokens', tokens)
            if len(tokens) == 0:
                print('raw_cat', raw_cat)
                continue

            tokens_new = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0:
                    tokens_new.append(glove_wordtoix[t])
            cat_labels.append(tokens_new)
            cat_label_lens.append(len(tokens_new))

    max_len = max(cat_label_lens)
    new_cat_labels = np.zeros((len(cat_labels), max_len), dtype=int)
    for i in range(len(cat_labels)):
        new_cat_labels[i,:cat_label_lens[i]] = np.array(cat_labels[i])

    cat_labels = torch.from_numpy(new_cat_labels)
    cat_label_lens = torch.LongTensor(cat_label_lens)

    sorted_cat_label_lens, sorted_cat_label_indices = \
        torch.sort(cat_label_lens, 0, True)

    sorted_cat_labels = cat_labels[sorted_cat_label_indices]
    _, resorted_cat_label_indices = torch.sort(sorted_cat_label_indices, 0, False)

    if cfg.CUDA:
        sorted_cat_labels = Variable(sorted_cat_labels).cuda()
        sorted_cat_label_lens = Variable(sorted_cat_label_lens).cuda()
        resorted_cat_label_indices = Variable(resorted_cat_label_indices).cuda()
    else:
        sorted_cat_labels = Variable(sorted_cat_labels)
        sorted_cat_label_lens = Variable(sorted_cat_label_lens)
        resorted_cat_label_indices = Variable(resorted_cat_label_indices)

    return sorted_cat_labels, sorted_cat_label_lens, resorted_cat_label_indices

def load_glove_vocab(dataset_path):
    TEXT = torchtext.data.Field(sequential=True, tokenize=en_tokenizer, lower=True)
    LABEL1 = torchtext.data.Field(sequential=True, use_vocab=False)
    LABEL2 = torchtext.data.Field(sequential=True, use_vocab=False)
    LABEL3 = torchtext.data.Field(sequential=True, use_vocab=False)
    LABEL4 = torchtext.data.Field(sequential=True, use_vocab=False)
    LABEL5 = torchtext.data.Field(sequential=True, use_vocab=False)

    tab_dataset = torchtext.data.TabularDataset(
        path=dataset_path, format='tsv',
        fields=[('TEXT', TEXT), ('LABEL1', LABEL1), ('LABEL2', LABEL2), 
            ('LABEL3', LABEL3), ('LABEL4', LABEL4), ('LABEL5', LABEL5)]
    )
    TEXT.build_vocab(tab_dataset, vectors="glove.6B.%dd"%(cfg.TEXT.GLOVE_EMBEDDING_DIM))

    return TEXT.vocab

def load_glove_emb(data_dir, split, train_names, test_names):
    filepath = os.path.join(data_dir, 'captions_glove.pickle')
    if not os.path.isfile(filepath):
        train_path = '%s/bbox_label/input_train2014.txt' % (data_dir)
        test_path = '%s/bbox_label/input_val2014.txt' % (data_dir)

        train_vocab = load_glove_vocab(train_path)
        test_vocab = load_glove_vocab(test_path)

        train_captions = load_captions(data_dir, train_names)
        test_captions = load_captions(data_dir, test_names)

        train_captions = translate_glove(train_captions, train_vocab.stoi)
        test_captions = translate_glove(test_captions, test_vocab.stoi)

        with open(filepath, 'wb') as f:
            pickle.dump([train_captions, test_captions,
                         train_vocab, test_vocab], f, protocol=2)
            print('Save to: ', filepath)
    else:
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            train_captions, test_captions = x[0], x[1]
            train_vocab, test_vocab = x[2], x[3]
            del x
            print('Load from: ', filepath)

    if split == 'train':
        # a list of list: each list contains
        # the indices of words in a sentence
        captions = train_captions
        glove_embed = nn.Embedding(len(train_vocab), cfg.TEXT.GLOVE_EMBEDDING_DIM)
        glove_embed.weight.data.copy_(train_vocab.vectors)
        ixtoword, wordtoix = train_vocab.itos, train_vocab.stoi
    else:  # split=='test'
        captions = test_captions
        glove_embed = nn.Embedding(len(test_vocab), cfg.TEXT.GLOVE_EMBEDDING_DIM)
        glove_embed.weight.data.copy_(test_vocab.vectors)
        ixtoword, wordtoix = test_vocab.itos, test_vocab.stoi

    return captions, ixtoword, wordtoix, glove_embed

################################### load image data #################################
def load_imgs_data(data_dir, split, filenames):
    filepath = os.path.join(data_dir, '%s_imgs.bigfile'%(split))
    if not os.path.isfile(filepath):
        print('writing %s imgs'%(split))
        write_imgs(data_dir, filenames, filepath)

    img_bytes = read_imgs(data_dir, filenames, filepath)

    return img_bytes


################################### load other data #################################
def load_class_id(data_dir, total_num):
    if os.path.isfile(data_dir + '/class_info.pickle'):
        with open(data_dir + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f)
    else:
        class_id = np.arange(total_num)
    return class_id

def load_filenames(data_dir, split):
    filepath = '%s/%s/filenames.pickle' % (data_dir, split)
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    else:
        filenames = []
    return filenames

def load_sample_filenames(data_dir):
    filepath = '%s/sample/filenames.txt' % (data_dir)
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            filenames_sentids = f.readlines()
        print('Load filenames from: %s (%d)' % (filepath, len(filenames_sentids)))
    else:
        filenames_sentids = []

    filenames_sentids = [name.replace("\r\n", "") for name in filenames_sentids]
    filenames, sentids = [], []
    for pair in filenames_sentids:
        filename, sentid = pair.split(',')
        filenames.append(filename)
        sentids.append(int(sentid))

    return filenames, sentids

def load_acts_data(data_dir, split):
    filepath = os.path.join(data_dir, '%s_acts_tf%d.pickle'%(split, cfg.TEST.USE_TF))
    if not os.path.isfile(filepath):
        print('Error: no such a file %s'%(filepath))
        return None
    else:
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            acts_dict = x[0]
            del x
            print('Load from: ', filepath)

    return acts_dict

################################### load anno data #################################
def load_anns_data(data_dir, split, postfix, ann_type, filenames, imsize, fmsize, cats_index_dict):
    # postfix: _gt_insanns.pickle or _gen_insanns.pickle
    filepath = os.path.join(data_dir, '%s%s'%(split, postfix))
    if not os.path.isfile(filepath):
        if ann_type == 'gt':
            insanns_dict = load_gt_insanns(data_dir, filenames, split, imsize, fmsize, cats_index_dict)
        elif ann_type == 'gen':
            insanns_dict = load_gen_insanns(data_dir, filenames, split, imsize, fmsize, cats_index_dict)
        
        with open(filepath, 'wb') as f:
            pickle.dump([insanns_dict], f, protocol=2)
            print('Save to: ', filepath)
    else:
        with open(filepath, 'rb') as f:
            x = pickle.load(f, encoding='latin1')
            insanns_dict = x[0]
            del x
            print('Load from: ', filepath)

    return insanns_dict


def load_gt_insanns(data_dir, filenames, split, imsize, fmsize, cats_index_dict):
    print('creating %s gt_insanns'%(split))
    if split == 'train':
        split_name = 'train'
    else:
        split_name = 'val'
    annFile = os.path.join(data_dir, 'insanns', 'instances_%s2014.json'%(split_name))
    coco = COCO(annFile)

    insanns_dict = {}
    
    for img_index in range(len(filenames)):
        if img_index % 500 == 0:
            print('%07d / %07d'%(img_index, len(filenames)))
        ### 1. initialize the ann containers
        anno_dict = {}
        rois = []
        for branch_index in range(cfg.TREE.BRANCH_NUM):
            rois.append(np.zeros(shape=(cfg.ROI.BOXES_NUM, cfg.ROI.BOXES_DIM)))
        fm_rois = np.zeros(shape=(cfg.ROI.BOXES_NUM, cfg.ROI.BOXES_DIM))

        ### 2. fetch image id and the corresponding annotation ids
        img_id = coco.getImgId(filenames[img_index])
        crowd_annIds, noncrowd_annIds = coco.getAnnIds(imgIds=img_id)

        ### 3. early finish if no annotations found
        if len(noncrowd_annIds) == 0:
            #anno_dict['hmaps'] = hmaps
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
        
        scales = np.zeros(shape=(cfg.TREE.BRANCH_NUM, 2))
        for branch_index in range(cfg.TREE.BRANCH_NUM):
            scales[branch_index, 0] = imsize[branch_index]/float(img_height)
            scales[branch_index, 1] = imsize[branch_index]/float(img_width)

        num_rois = len(noncrowd_anns)
        raw_rois = np.zeros((num_rois, 6))
        kept_indices = []
        count = 0
        for roi_index in range(num_rois):
            roi = noncrowd_anns[roi_index]['bbox']
            bbox_width, bbox_height = roi[2:4]
            scaled_width = scales[cfg.TREE.BRANCH_NUM-1, 1]*bbox_width
            scaled_height = scales[cfg.TREE.BRANCH_NUM-1, 0]*bbox_height

            if scaled_width < cfg.ROI.ROI_MIN_SIZE and scaled_height < cfg.ROI.ROI_MIN_SIZE:
                continue

            kept_indices.append(roi_index)

            raw_rois[count, :4] = roi
            raw_rois[count, 4] = cats_index_dict[noncrowd_anns[roi_index]['category_id']]
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
        for branch_index in range(cfg.TREE.BRANCH_NUM):
            rois[branch_index][:num_rois, :] = raw_rois
            rois[branch_index][:, [0, 2]] = rois[branch_index][:, [0, 2]]*scales[branch_index, 1]
            rois[branch_index][:, [1, 3]] = rois[branch_index][:, [1, 3]]*scales[branch_index, 0]

        fm_rois[:num_rois, :] = rois[0][:num_rois, :].copy()
        fm_rois[:, :4] = fm_rois[:, :4] / 2.0

        ### 9. construct seg masks
        raw_masks = []
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
            re_mask = resize(mask, [imsize[0], imsize[0]])
            raw_masks.append(re_mask)
        raw_pooled_masks = np.amax(raw_masks, axis=0)

        ### 10. construct bbox maps
        raw_bbox_masks = np.zeros((num_rois, imsize[0], imsize[0]))
        for roi_index in range(num_rois):
            x_start = min(int(round(rois[0][roi_index, 0])), imsize[0]-1)
            y_start = min(int(round(rois[0][roi_index, 1])), imsize[0]-1)
            x_end = min(int(round(rois[0][roi_index, 0]+rois[0][roi_index, 2])), imsize[0]-1)
            y_end = min(int(round(rois[0][roi_index, 1]+rois[0][roi_index, 3])), imsize[0]-1)
            raw_bbox_masks[roi_index, y_start:y_end, x_start:x_end] = 1

        ### 11. construct bbox maps
        raw_bbox_fmasks = np.zeros((num_rois, fmsize, fmsize))
        for roi_index in range(num_rois):
            x_start = min(int(round(fm_rois[roi_index, 0]/2.0)), fmsize-1) # 32 -> 16
            y_start = min(int(round(fm_rois[roi_index, 1]/2.0)), fmsize-1)
            x_end = min(int(round(fm_rois[roi_index, 0]/2.0+fm_rois[roi_index, 2]/2.0)), fmsize-1)
            y_end = min(int(round(fm_rois[roi_index, 1]/2.0+fm_rois[roi_index, 3]/2.0)), fmsize-1)
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

def load_gen_insanns(data_dir, filenames, split, imsize, fmsize, cats_index_dict):
    print('creating %s gen_insanns'%(split))
    gen_dir = '%s/gen_masks/'%(data_dir)
    insanns_dict = {}
    
    for img_index in range(len(filenames)):
        if img_index % 500 == 0:
            print('%07d / %07d'%(img_index, len(filenames)))
        ### 1. initialize the ann containers
        anno_dicts = {}

        ### 2. load gen_boxes
        gen_bbox_paths = glob('%s%s/*'%(gen_dir, filenames[img_index]))
        gen_bbox_paths_indices = [int(path_leaf(gen_bbox_path)) for gen_bbox_path in gen_bbox_paths]
        gen_bbox_paths_indices2 = np.argsort(gen_bbox_paths_indices)
        gen_bbox_paths = [gen_bbox_paths[index] for index in gen_bbox_paths_indices2]
        gen_bbox_paths_indices.sort()

        gen_boxes_set = []
        for gen_bbox_path in gen_bbox_paths:
            sub_gen_bbox_path = '%s/boxes.txt'%(gen_bbox_path)
            if is_non_zero_file(sub_gen_bbox_path):
                gen_boxes = pd.read_csv(sub_gen_bbox_path, header=None).astype(int)
                gen_boxes = np.array(gen_boxes)
                gen_boxes_set.append(gen_boxes)
            else:
                gen_boxes_set.append(None)

        scales = np.zeros(shape=(cfg.TREE.BRANCH_NUM, 2))
        for branch_index in range(cfg.TREE.BRANCH_NUM):
            scales[branch_index, 0] = imsize[branch_index]/float(imsize[-1])
            scales[branch_index, 1] = imsize[branch_index]/float(imsize[-1])

        for boxes_index in range(len(gen_bbox_paths_indices)):
            gen_bbox_paths_index = gen_bbox_paths_indices[boxes_index]
            anno_dict = {}
            rois = []
            for branch_index in range(cfg.TREE.BRANCH_NUM):
                rois.append(np.zeros(shape=(cfg.ROI.BOXES_NUM, cfg.ROI.BOXES_DIM)))
            fm_rois = np.zeros(shape=(cfg.ROI.BOXES_NUM, cfg.ROI.BOXES_DIM))

            if gen_boxes_set[boxes_index] is None:
                anno_dict['rois'] = rois
                anno_dict['fm_rois'] = fm_rois
                anno_dict['bbox maps'] = None
                anno_dict['bbox fmaps'] = None
                anno_dict['num_rois'] = 0
                anno_dicts[gen_bbox_paths_index] = anno_dict
                continue

            ### 5. filter small annotations
            num_rois = gen_boxes_set[boxes_index].shape[0]
            raw_rois = np.zeros((num_rois, 6))
            kept_indices = []
            count = 0
            for roi_index in range(num_rois):
                roi = gen_boxes_set[boxes_index][roi_index]
                bbox_width, bbox_height = roi[2:4]

                if bbox_width < cfg.ROI.ROI_MIN_SIZE and bbox_height < cfg.ROI.ROI_MIN_SIZE:
                    continue

                kept_indices.append(roi_index)
                raw_rois[count, :4] = roi[:4]
                raw_rois[count, 4] = cats_index_dict[roi[4]]
                count += 1

            num_rois = len(kept_indices)
            raw_rois = raw_rois[:num_rois]

            ### 6. early finish if no annotations left
            if num_rois == 0:
                anno_dict['rois'] = rois
                anno_dict['fm_rois'] = fm_rois
                anno_dict['bbox maps'] = None
                anno_dict['bbox fmaps'] = None
                anno_dict['num_rois'] = 0
                anno_dicts[gen_bbox_paths_index] = anno_dict
                continue

            ### 7. refine annotations according to the kept_indices
            if num_rois > cfg.ROI.BOXES_NUM:
                raw_rois, sorted_indices = calc_sort_size(raw_rois)
                raw_rois = raw_rois[:cfg.ROI.BOXES_NUM,:]
                kept_indices = sorted_indices[:cfg.ROI.BOXES_NUM]
                num_rois = cfg.ROI.BOXES_NUM

            ### 8. assemble the rois container
            for branch_index in range(cfg.TREE.BRANCH_NUM):
                rois[branch_index][:num_rois, :] = raw_rois
                rois[branch_index][:, [0, 2]] = rois[branch_index][:, [0, 2]]*scales[branch_index, 1]
                rois[branch_index][:, [1, 3]] = rois[branch_index][:, [1, 3]]*scales[branch_index, 0]

            fm_rois[:num_rois, :] = rois[0][:num_rois, :].copy()
            fm_rois[:, :4] = fm_rois[:, :4] / 2.0

            ### 10. construct bbox maps
            raw_bbox_masks = np.zeros((num_rois, imsize[0], imsize[0]))
            for roi_index in range(num_rois):
                x_start = min(int(round(rois[0][roi_index, 0])), imsize[0]-1)
                y_start = min(int(round(rois[0][roi_index, 1])), imsize[0]-1)
                x_end = min(int(round(rois[0][roi_index, 0]+rois[0][roi_index, 2])), imsize[0]-1)
                y_end = min(int(round(rois[0][roi_index, 1]+rois[0][roi_index, 3])), imsize[0]-1)
                raw_bbox_masks[roi_index, y_start:y_end, x_start:x_end] = 1

            ### 11. construct bbox maps
            raw_bbox_fmasks = np.zeros((num_rois, fmsize, fmsize))
            for roi_index in range(num_rois):
                x_start = min(int(round(fm_rois[roi_index, 0]/2.0)), fmsize-1) # 32 -> 16
                y_start = min(int(round(fm_rois[roi_index, 1]/2.0)), fmsize-1)
                x_end = min(int(round(fm_rois[roi_index, 0]/2.0+fm_rois[roi_index, 2]/2.0)), fmsize-1)
                y_end = min(int(round(fm_rois[roi_index, 1]/2.0+fm_rois[roi_index, 3]/2.0)), fmsize-1)
                raw_bbox_fmasks[roi_index, y_start:y_end, x_start:x_end] = 1

            anno_dict['rois'] = rois
            anno_dict['fm_rois'] = fm_rois
            anno_dict['bbox maps'] = raw_bbox_masks
            anno_dict['bbox fmaps'] = raw_bbox_fmasks
            anno_dict['num_rois'] = num_rois
            anno_dicts[gen_bbox_paths_index] = anno_dict

        insanns_dict[filenames[img_index]] = anno_dicts

    return insanns_dict