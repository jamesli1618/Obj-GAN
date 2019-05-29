from __future__ import print_function, division

import torch
#import torchtext

import seq2seq
from seq2seq.dataset.prepare_dataset import random_batch
import sys
import numpy as np
import collections
import os

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, batch_size, early_stop_len, expt_dir, dev_cap_lang, dev_label_lang, 
        x_mean_std, y_mean_std, w_mean_std, r_mean_std, gaussian_dict, box_saving_folder,
        output_opt):
        self.batch_size = batch_size
        self.early_stop_len = early_stop_len
        self.expt_dir = expt_dir
        self.dev_cap_lang = dev_cap_lang
        self.dev_label_lang = dev_label_lang
        self.x_mean_std = x_mean_std
        self.y_mean_std = y_mean_std
        self.w_mean_std = w_mean_std
        self.r_mean_std = r_mean_std
        self.gaussian_dict = gaussian_dict
        self.display_step = 200 #500
        self.box_saving_folder = box_saving_folder
        self.std_img_size = 256.0
        self.output_opt = output_opt

    def evaluate(self, encoder, decoder, data, keys):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        # create a recording dictionary for keys
        keys_dict = {}
        for key in keys:
            if key not in keys_dict:
                keys_dict[key] = 0

        index2word = self.dev_cap_lang.index2word

        if self.output_opt == 1:
            fout_bbox_label = open('%s/dev_bbox_test.txt'%(self.expt_dir), 'w')
        steps_per_epoch = int(round(len(data)/self.batch_size))
        for batch_index in range(steps_per_epoch): #steps_per_epoch
            if (batch_index+1) % self.display_step == 0:
                print('%07d / %07d'%(batch_index, steps_per_epoch))
                
            input_variables, input_lengths, target_l_variables, target_lengths, target_x_variables, target_y_variables, \
                target_w_variables, target_h_variables = random_batch(self.batch_size, data, self.dev_cap_lang, 
                self.dev_label_lang, self.x_mean_std, self.y_mean_std, self.w_mean_std, self.r_mean_std, 
                is_training=0, select_index=batch_index)

            encoder_outputs, encoder_hidden = encoder(input_variables, input_lengths)
            decoder_outputs, xy_gmm_params, wh_gmm_params, decoder_hidden, other = \
                decoder(encoder_hidden, encoder_outputs, target_l_variables, target_x_variables, 
                    target_y_variables, target_w_variables, target_h_variables, is_training=0, 
                    early_stop_len=self.early_stop_len)

            xs, ys = self.coord_converter(other['xy'], self.x_mean_std[0], self.x_mean_std[1], self.y_mean_std[0], self.y_mean_std[1])
            ws, hs = self.coord_converter(other['wh'], self.w_mean_std[0], self.w_mean_std[1], self.r_mean_std[0], self.r_mean_std[1])
            hs = np.multiply(ws, hs)
            ls = torch.cat(other['sequence']).view(-1,self.batch_size).transpose(0,1)
            ls = np.array(ls.cpu().data.tolist())[0]
            xs, ys, ws, hs, ls = self.validity_indices(xs, ys, ws, hs, ls)

            if self.output_opt == 0:
                # create folders if not exist
                key_dir = self.box_saving_folder + keys[batch_index]
                if not os.path.exists(key_dir):
                    os.makedirs(key_dir)

                # get the index of the caption for the current key, and increase the record
                cur_index = keys_dict[keys[batch_index]]
                keys_dict[keys[batch_index]] = keys_dict[keys[batch_index]]+1
                key_sub_dir = key_dir + '/%d/'%(cur_index)
                if not os.path.exists(key_sub_dir):
                    os.makedirs(key_sub_dir)

                fout_filename = open('%s/boxes.txt'%(key_sub_dir), 'w')

            if len(ls) > 1:
                xs = xs[:-1]
                ys = ys[:-1]
                ws = ws[:-1]
                hs = hs[:-1]
                ls = ls[:-1]
                ls = [int(self.dev_label_lang.index2word[int(l)]) for l in ls]
                ls = np.array(ls)

                # filter redundant labels
                counter = collections.Counter(ls)
                unique_labels, label_counts = list(counter.keys()), list(counter.values())
                kept_indices = []
                for label_index in range(len(unique_labels)):
                    label = unique_labels[label_index]
                    label_num = label_counts[label_index]
                    # sample an upper-bound threshold for this label
                    mu, sigma = self.gaussian_dict[label]
                    threshold = max(int(np.random.normal(mu, sigma, 1)), 2)
                    old_indices = np.where(ls == label)[0].tolist()
                    new_indices = old_indices
                    if threshold < len(old_indices):
                        new_indices = old_indices[:threshold]
                    kept_indices += new_indices

                kept_indices.sort()
                xs = xs[kept_indices]
                ys = ys[kept_indices]
                ws = ws[kept_indices]
                hs = hs[kept_indices]
                ls = ls[kept_indices]
                ls = [str(l) for l in ls]

                xs = xs - ws/2.0
                xs = np.clip(xs, 1, self.std_img_size-1)
                ys = ys - hs/2.0
                ys = np.clip(ys, 1, self.std_img_size-1)
                ws = np.minimum(ws, self.std_img_size-xs)
                hs = np.minimum(hs, self.std_img_size-ys)


                if self.output_opt == 0:
                    for i in range(len(ls)):
                        fout_filename.write('%.2f,%.2f,%.2f,%.2f,%s,0\n'%(xs[i], ys[i], ws[i], hs[i], ls[i]))


                if self.output_opt == 1:
                    cap_seq = input_variables[0].tolist()
                    cap_seq = [index2word[word] for word in cap_seq]

                    fout_bbox_label.write('%s - %s - '%(keys[batch_index], cap_seq))
                    for i in range(len(ls)):
                        fout_bbox_label.write('%.2f,%.2f,%.2f,%.2f,%s - '%(xs[i], ys[i], ws[i], hs[i], ls[i]))
                    fout_bbox_label.write('\n')

            if self.output_opt == 0:
                fout_filename.close()
        if self.output_opt == 1:
            fout_bbox_label.close()


    def coord_converter(self, coord_seq, mean_x, std_x, mean_y, std_y):
        coord_x_seq, coord_y_seq = [], []
        for i in range(len(coord_seq)):
            x, y = coord_seq[i]
            coord_x_seq.append(x*std_x+mean_x)
            coord_y_seq.append(y*std_y+mean_y)

        return np.array(coord_x_seq), np.array(coord_y_seq)

    def validity_indices(self, x_seq, y_seq, w_seq, h_seq, l_seq):
        x_valid_indices = x_seq > 0
        y_valid_indices = y_seq > 0
        w_valid_indices = w_seq > 0
        h_valid_indices = h_seq > 0
        valid_indices = np.multiply(np.multiply(np.multiply(x_valid_indices, y_valid_indices), w_valid_indices), h_valid_indices)
        x_seq = x_seq[valid_indices]
        y_seq = y_seq[valid_indices]
        w_seq = w_seq[valid_indices]
        h_seq = h_seq[valid_indices]
        l_seq = l_seq[valid_indices]

        return x_seq, y_seq, w_seq, h_seq, l_seq
