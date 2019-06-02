import sys
import unicodedata
import string
import re
import numpy as np
import random
import collections
from torch.autograd import Variable
import torch
from nltk.tokenize import RegexpTokenizer

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.word2count = {"<pad>": 0, "<sos>": 0, "<eos>": 0, "<unk>": 0}
        self.index2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.n_words = 4 # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_cap_words(self, sentence):
        for word in sentence:
            self.index_cap_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def index_cap_word(self, word):
        if word in self.word2index:
            self.word2count[word] += 1

    def reset_word2count(self):
        self.word2count = {key: 0 for key in self.word2index}

    def enrich_word2count(self):
        for word in self.word2index:
            if word not in self.word2count:
                self.word2count[word] = 0

    def reset_ext_word2count(self):
        self.word2count = {}
        for word in self.word2index:
            self.word2count[word] = 0

    def copy_dict(self, lang):
        self.word2index = lang.word2index
        self.word2count = lang.word2count
        self.index2word = lang.index2word
        self.n_words = lang.n_words
        self.reset_word2count()

    def copy_ext_dict(self, ixtoword, wordtoix):
        self.word2index = wordtoix
        self.index2word = ixtoword
        self.n_words = len(self.word2index)

        self.word2count = {}
        self.reset_ext_word2count()

    def set_word2index(self, word2index):
        self.word2index = word2index
        self.n_words = len(self.word2index)

    def set_index2word(self, index2word):
        self.index2word = index2word
        
    def increase_seos_count(self):
        self.word2count["<sos>"] += 1
        self.word2count["<eos>"] += 1

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    cap = s.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(cap.lower())

    tokens_new = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0:
            tokens_new.append(t)
    return tokens_new

def read_langs(filename):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(filename).read().strip().split('\n')

    # Split every line into tuples and normalize captions
    tuples, xs, ys, ws, hs = [], [], [], [], []
    line_num = 0
    for line in lines:
        if line_num % 50000 == 0:
            print('loading {}/{}'.format(line_num, len(lines)))
        str_cap, str_x, str_y, str_w, str_h, str_label = line.split('\t')
        str_cap = normalize_string(str_cap)
        tuples.append([str_cap, str_x, str_y, str_w, str_h, str_label])
        line_num += 1

    cap_lang = Lang('caption')
    label_lang = Lang('label')

    return cap_lang, label_lang, tuples

def filter_tuples(tuples, max_len, min_len):
    filtered_tuples = []
    for item in tuples:
        if len(item[0]) >= min_len and len(item[0]) <= max_len \
            and len(item[5]) >= min_len and len(item[5]) <= max_len:
                filtered_tuples.append(item)
    return filtered_tuples

def read_mean_std(filename):
    lines = open(filename).read().strip().split('\n')
    x_mean, x_std = [float(val) for val in lines[0].split(' ')]
    y_mean, y_std = [float(val) for val in lines[1].split(' ')]
    w_mean, w_std = [float(val) for val in lines[2].split(' ')]
    r_mean, r_std = [float(val) for val in lines[3].split(' ')]

    return (x_mean, x_std), (y_mean, y_std), (w_mean, w_std), (r_mean, r_std)

def prepare_data(train_path, dev_path, mean_std_path, max_len, min_len, ixtoword, wordtoix):
    print('len(ixtoword): ', len(ixtoword))
    print("Reading means and stds")
    x_mean_std, y_mean_std, w_mean_std, r_mean_std = read_mean_std(mean_std_path)

    train_cap_lang, train_label_lang, train_tuples = read_langs(train_path)
    print("Read %d training sentence tuples" % len(train_tuples))

    train_cap_lang.copy_ext_dict(ixtoword, wordtoix)

    print("Indexing training words...")
    for item in train_tuples:
        train_cap_lang.index_cap_words(item[0])
        train_label_lang.index_words(item[5])
        train_label_lang.increase_seos_count()
    
    print('Indexed %d training words in captions, %d labels' % (
        train_cap_lang.n_words, train_label_lang.n_words))

    dev_cap_lang, dev_label_lang, dev_tuples = read_langs(dev_path)
    print("Read %d dev sentence tuples" % len(dev_tuples))

    print("Indexing dev words...")
    dev_cap_lang.copy_dict(train_cap_lang)
    dev_label_lang.copy_dict(train_label_lang)

    for item in dev_tuples:
        dev_cap_lang.index_cap_words(item[0])
        dev_label_lang.index_words(item[5])
        dev_label_lang.increase_seos_count()

    print('Indexed %d dev words in captions, %d labels' % (
        dev_cap_lang.n_words, dev_label_lang.n_words))

    return train_cap_lang, train_label_lang, train_tuples, dev_cap_lang, \
    dev_label_lang, dev_tuples, x_mean_std, y_mean_std, w_mean_std, r_mean_std

def prepare_test_data(dev_path, mean_std_path, max_len, min_len, 
    train_cap_word2index, train_cap_index2word, train_label_word2index, 
    train_label_index2word, dev_filename_path):
    print('Reading img keys:')
    keys = open(dev_filename_path).read().strip().split('\n')

    print("Reading means and stds")
    x_mean_std, y_mean_std, w_mean_std, r_mean_std = read_mean_std(mean_std_path)

    dev_cap_lang, dev_label_lang, dev_tuples = read_langs(dev_path)
    print("Read %d dev sentence tuples" % len(dev_tuples))

    print("Indexing dev words...")
    dev_cap_lang.set_word2index(train_cap_word2index)
    dev_cap_lang.set_index2word(train_cap_index2word)
    dev_cap_lang.reset_word2count()
    dev_label_lang.set_word2index(train_label_word2index)
    dev_label_lang.set_index2word(train_label_index2word)
    dev_label_lang.reset_word2count()

    print('Indexed %d dev words in captions, %d labels' % (
        dev_cap_lang.n_words, dev_label_lang.n_words))

    return dev_cap_lang, dev_label_lang, dev_tuples, x_mean_std, \
    y_mean_std, w_mean_std, r_mean_std, keys

def get_class_sta(train_path, gaussian_dict_path):
    train_cap_lang, train_label_lang, train_tuples = read_langs(train_path)
    print("Read %d training sentence tuples" % len(train_tuples))

    sta_dict = {}
    gaussian_dict = {}
    for item in train_tuples:
        labels = [int(label_str) for label_str in item[5].split(' ')]
        counter = collections.Counter(labels)
        unique_labels, label_counts = list(counter.keys()), list(counter.values())
        for label_index in range(len(unique_labels)):
            label = unique_labels[label_index]
            count = label_counts[label_index]
            if label not in sta_dict:
                sta_dict[label] = []
                sta_dict[label].append(count)
            else:
                sta_dict[label].append(count)

    for label in sta_dict:
        tmp_mean = np.mean(np.array(sta_dict[label]))
        tmp_std = np.std(np.array(sta_dict[label]))
        gaussian_dict[label] = (tmp_mean, tmp_std)
    np.save(gaussian_dict_path, gaussian_dict)

# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence):
    if "<sos>" in lang.word2index:
        seq = [lang.word2index["<sos>"]]
    else:
        seq = []
    if type(sentence) is list:
        words = sentence
    else:
        words = sentence.split(' ')
    for word in words:
        if word in lang.word2index:
            seq.append(lang.word2index[word])
    if "<eos>" in lang.word2index:
        seq.append(lang.word2index["<eos>"])
    return seq

def nums_from_sentence(mean, std, sentence):
    return [0] + [(float(num)-mean)/std for num in sentence.split(' ')] + [0]

# Pad a with the PAD symbol
def pad_seq(seq, max_length, pad_token):
    seq += [pad_token for i in range(max_length - len(seq))]
    return seq

def random_batch(batch_size, tuples, cap_lang, label_lang, x_mean_std, 
    y_mean_std, w_mean_std, r_mean_std, is_training=0, select_index=None):
    cap_seqs, label_seqs, x_seqs, y_seqs, w_seqs, r_seqs = [], [], [], [], [], []
    if is_training:
        # Choose random tuples
        for i in range(batch_size):
            item = random.choice(tuples)
            cap_seqs.append(indexes_from_sentence(cap_lang, item[0]))
            label_seqs.append(indexes_from_sentence(label_lang, item[5]))
            x_seqs.append(nums_from_sentence(x_mean_std[0], x_mean_std[1], item[1]))
            y_seqs.append(nums_from_sentence(y_mean_std[0], y_mean_std[1], item[2]))
            w_seqs.append(nums_from_sentence(w_mean_std[0], w_mean_std[1], item[3]))
            r_seqs.append(nums_from_sentence(r_mean_std[0], r_mean_std[1], item[4]))
    else:
        assert select_index is not None
        item = tuples[select_index]
        cap_seqs.append(indexes_from_sentence(cap_lang, item[0]))
        label_seqs.append(indexes_from_sentence(label_lang, item[5]))
        x_seqs.append(nums_from_sentence(x_mean_std[0], x_mean_std[1], item[1]))
        y_seqs.append(nums_from_sentence(y_mean_std[0], y_mean_std[1], item[2]))
        w_seqs.append(nums_from_sentence(w_mean_std[0], w_mean_std[1], item[3]))
        r_seqs.append(nums_from_sentence(r_mean_std[0], r_mean_std[1], item[4]))

    # Zip into pairs, sort by length (descending), unzip
    seq_tuples = sorted(zip(cap_seqs, label_seqs, x_seqs, y_seqs, w_seqs, r_seqs), 
        key=lambda p: len(p[0]), reverse=True)
    cap_seqs, label_seqs, x_seqs, y_seqs, w_seqs, r_seqs = zip(*seq_tuples)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    cap_lengths = [len(s) for s in cap_seqs]
    cap_padded = [pad_seq(s, max(cap_lengths), 0) for s in cap_seqs]
    label_lengths = [len(s) for s in label_seqs]
    label_padded = [pad_seq(s, max(label_lengths), label_lang.word2index["<pad>"]) 
    for s in label_seqs]
    x_padded = [pad_seq(s, max(label_lengths), 0) for s in x_seqs]
    y_padded = [pad_seq(s, max(label_lengths), 0) for s in y_seqs]
    w_padded = [pad_seq(s, max(label_lengths), 0) for s in w_seqs]
    r_padded = [pad_seq(s, max(label_lengths), 0) for s in r_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    cap_var = Variable(torch.LongTensor(cap_padded)).cuda()
    label_var = Variable(torch.LongTensor(label_padded)).cuda()
    x_var = Variable(torch.FloatTensor(x_padded)).cuda()
    y_var = Variable(torch.FloatTensor(y_padded)).cuda()
    w_var = Variable(torch.FloatTensor(w_padded)).cuda()
    r_var = Variable(torch.FloatTensor(r_padded)).cuda()
        
    return cap_var, cap_lengths, label_var, label_lengths, x_var, y_var, w_var, r_var