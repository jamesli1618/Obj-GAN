from __future__ import division
import logging
import os
import random
import time

import torch
#import torchtext
from torch import optim

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import Perplexity, BBLoss
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset.prepare_dataset import random_batch

import sys

class SupervisedTrainer(object):
    def __init__(self, expt_dir='experiment', lloss=None, bloss=None, batch_size=64,
        random_seed=None, checkpoint_every=100, print_every=1, train_cap_lang=None, 
        train_label_lang=None, x_mean_std=None, y_mean_std=None, w_mean_std=None, 
        r_mean_std=None):
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.lloss = lloss
        self.bloss = bloss
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

        self.train_cap_lang = train_cap_lang
        self.train_label_lang = train_label_lang
        self.x_mean_std = x_mean_std
        self.y_mean_std = y_mean_std
        self.w_mean_std = w_mean_std
        self.r_mean_std = r_mean_std

    def _train_batch(self, input_variable, input_lengths, target_l_variables, 
        target_x_variables, target_y_variables, target_w_variables, target_h_variables, 
        encoder, decoder, is_training, batch_step):
        lloss = self.lloss
        bloss = self.bloss
        # input_variable: batch x max_input_len
        # target_l_variables: batch x max_target_len
        # target_x_variables: batch x max_target_len
        # Forward propagation
        encoder_outputs, encoder_hidden = encoder(input_variable, input_lengths)
        encoder_outputs = encoder_outputs.detach()
        encoder_hidden = tuple([h.detach() for h in encoder_hidden])
        decoder_outputs, xy_gmm_params, wh_gmm_params, decoder_hidden, other = \
            decoder(encoder_hidden, encoder_outputs, target_l_variables, 
                target_x_variables, target_y_variables, target_w_variables, 
                target_h_variables, is_training=is_training)
        if batch_step % self.print_every == 0:
            l_seqlist = other['sequence']
            l_seqlist2 = torch.cat(l_seqlist).view(-1,self.batch_size).transpose(0,1)
            human_label = self.train_label_lang.word2index[str(1)]
            num_not_human = torch.sum(l_seqlist2 != human_label)
            l_match = 0
            total = 0
            pad = self.train_label_lang.word2index["<pad>"]

        # Get loss
        lloss.reset()
        bloss.reset()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_l_variables.size(0)

            target_l = target_l_variables[:, step + 1]
            target_x = target_x_variables[:, step + 1]
            target_y = target_y_variables[:, step + 1]
            target_w = target_w_variables[:, step + 1]
            target_h = target_h_variables[:, step + 1]

            lloss.eval_batch(step_output.contiguous().view(batch_size, -1), target_l)
            bloss.eval_batch(xy_gmm_params[step], wh_gmm_params[step], target_x,
                target_y, target_w, target_h, target_l)

            if batch_step % self.print_every == 0:
                non_padding = target_l.ne(pad)
                l_correct = l_seqlist[step].view(-1).eq(target_l).masked_select(
                    non_padding).sum().item()
                l_match += l_correct
                total += non_padding.sum().item()

        # Backward propagation
        cur_lloss = lloss.get_loss()
        cur_bloss = bloss.get_loss()
        loss = cur_lloss+cur_bloss
        decoder.zero_grad()
        loss.backward()
        self.optimizer.step()

        if batch_step % self.print_every == 0:
            if total == 0:
                l_accuracy = float('nan')
            else:
                l_accuracy = l_match / total
    
            print('l_accuracy: {}'.format(l_accuracy))

        return cur_lloss.item(), cur_bloss.item()

    def _train_epoches(self, data, encoder, decoder, n_epochs, start_epoch, start_step,
                       dev_data=None, is_training=0):
        log = self.logger

        print_lloss_total = 0
        print_bloss_total = 0
        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        steps_per_epoch = int(round(len(data)/self.batch_size))
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            print('epoch: ', epoch)
            decoder.train(True)

            for batch_index in range(steps_per_epoch):
                step += 1
                step_elapsed += 1

                input_variables, input_lengths, target_l_variables, target_lengths, \
                target_x_variables, target_y_variables, target_w_variables, \
                target_h_variables = random_batch(self.batch_size, data, self.train_cap_lang, 
                    self.train_label_lang, self.x_mean_std, self.y_mean_std, self.w_mean_std, 
                    self.r_mean_std, is_training=1)

                lloss, bloss = self._train_batch(input_variables, input_lengths, 
                    target_l_variables, target_x_variables, target_y_variables, 
                    target_w_variables, target_h_variables, encoder, decoder, is_training, step)

                # Record average loss
                print_lloss_total += lloss
                print_bloss_total += bloss
                print_loss_total += lloss+bloss
                epoch_loss_total += lloss+bloss

                if step % self.print_every == 0:
                    print('step: ', step)
                    print_lloss_avg = print_lloss_total / self.print_every
                    print_bloss_avg = print_bloss_total / self.print_every
                    print_loss_avg = print_loss_total / self.print_every
                    print_lloss_total = 0
                    print_bloss_total = 0
                    print_loss_total = 0
                    log_msg = '%d/%d Progress: %d%%, Train %s: %.4f, %s: %.4f, %s: %.4f' % (
                        step,
                        steps_per_epoch,
                        step / total_steps * 100,
                        'Total',
                        print_loss_avg,
                        self.lloss.name,
                        print_lloss_avg,
                        self.bloss.name,
                        print_bloss_avg)
                    log.info(log_msg)

                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:
                    Checkpoint(model=decoder,
                        optimizer=self.optimizer,
                        epoch=epoch, step=step,
                        cap_word2index=self.train_cap_lang.word2index,
                        cap_index2word=self.train_cap_lang.index2word,
                        label_word2index=self.train_label_lang.word2index,
                        label_index2word=self.train_label_lang.index2word).save(self.expt_dir)

            log.info(log_msg)

    def train(self, encoder, decoder, data, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None, is_training=0):
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            decoder = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(decoder.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(decoder.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, 
            self.optimizer.scheduler))

        self._train_epoches(data, encoder, decoder, num_epochs, start_epoch, step, 
            dev_data=dev_data, is_training=is_training)
        return decoder
