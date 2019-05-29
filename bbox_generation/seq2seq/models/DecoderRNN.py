import random
import sys
import numpy as np
from random import randint

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as distributions

from .attention import Attention
from .baseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(BaseRNN):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    KEY_XYS = 'xy'
    KEY_WHS = 'wh'

    def __init__(self, l_word2index, x_mean, y_mean, w_mean, r_mean, batch_size, max_len, 
        hidden_size, gmm_comp_num, n_layers=1, rnn_cell='lstm', bidirectional=False,
        input_dropout_p=0, dropout_p=0, use_attention=False):
        super(DecoderRNN, self).__init__(len(l_word2index), max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.l_output_size = len(l_word2index)
        self.aug_size = 50

        self.batch_size = batch_size
        self.max_length = max_len
        # gmm_comp_num: number of gaussian mixture components
        self.gmm_comp_num = gmm_comp_num
        self.gmm_param_num = 6 # pi, u_x, u_y, sigma_x, sigma_y, rho_xy
        self.clip_val = 0.01
        self.std_norm = distributions.Normal(torch.Tensor([[0.0]*2]*self.batch_size), 
            torch.Tensor([[1.0]*2]*self.batch_size))
        self.use_attention = use_attention
        self.l_eos_id = l_word2index["<eos>"]
        self.l_sos_id = l_word2index["<sos>"]
        self.x_mean = x_mean
        self.y_mean = y_mean
        self.w_mean = w_mean
        self.r_mean = r_mean
        self.temperature = 0.4

        self.init_input = None
        self.l_embedding = nn.Embedding(self.l_output_size, self.hidden_size)
        self.xy_embedding = nn.Linear(2, self.aug_size)
        self.xy_input_dropout = nn.Dropout(p=self.input_dropout_p)
        self.wh_embedding = nn.Linear(2, self.aug_size)
        self.wh_input_dropout = nn.Dropout(p=self.input_dropout_p)
        self.next_xy_embedding = nn.Linear(2, self.aug_size)
        self.next_xy_input_dropout = nn.Dropout(p=self.input_dropout_p)
        if use_attention:
            self.attention = Attention('concat', self.hidden_size)
            # rnn inputs: input_size (l_output_size+4), hidden_size, num_layers
            self.rnn = self.rnn_cell(2*hidden_size+2*self.aug_size, hidden_size, 
                n_layers, batch_first=True, dropout=dropout_p)
        else:
            # rnn inputs: input_size (l_output_size+4), hidden_size, num_layers
            self.rnn = self.rnn_cell(hidden_size+2*self.aug_size, hidden_size, 
                n_layers, batch_first=True, dropout=dropout_p)

        self.l_softmax = F.log_softmax
        self.l_out = nn.Linear(self.hidden_size, self.l_output_size)
        #1*gmm_comp_num for pi, 2*gmm_comp_num for u, 3*gmm_comp_num for lower triangular matrix
        self.xy_out = nn.Linear(self.hidden_size+self.l_output_size, 
            self.gmm_comp_num*self.gmm_param_num)
        self.wh_out = nn.Linear(self.hidden_size+self.l_output_size+self.aug_size, 
            self.gmm_comp_num*self.gmm_param_num)

    def forward_step(self, l_decoder_input, x_decoder_input, y_decoder_input, 
        w_decoder_input, h_decoder_input, hidden, encoder_outputs, next_l_decoder_input=None, 
        next_x_decoder_input=None, next_y_decoder_input=None, is_training=0):
        batch_size = l_decoder_input.size(0)

        ### 1. get the RNN input ###
        # l_decoder_input: batch x output_size (1)
        l_decoder_input = l_decoder_input.unsqueeze(1)
        x_decoder_input = x_decoder_input.unsqueeze(1)
        y_decoder_input = y_decoder_input.unsqueeze(1)
        w_decoder_input = w_decoder_input.unsqueeze(1)
        h_decoder_input = h_decoder_input.unsqueeze(1)

        output_size = 1
        l_decoder_input = self.l_embedding(l_decoder_input)
        l_decoder_input = self.input_dropout(l_decoder_input)

        xy_decoder_input = self.xy_embedding(torch.cat((x_decoder_input, y_decoder_input), dim=1))
        xy_decoder_input = self.xy_input_dropout(xy_decoder_input)
        xy_decoder_input = xy_decoder_input.unsqueeze(0)
        wh_decoder_input = self.wh_embedding(torch.cat((w_decoder_input, h_decoder_input), dim=1))
        wh_decoder_input = self.wh_input_dropout(wh_decoder_input)
        wh_decoder_input = wh_decoder_input.unsqueeze(0)

        attn = None
        if self.use_attention:
            # encoder_outputs: batch x in_seq_len x hidden_size
            # hidden[0]: output_size (1) x batch x hidden_size
            # attn: batch x output_size x in_seq_len
            attn = self.attention(hidden[0], encoder_outputs)
            # context: batch x output_size x hidden_size
            context = attn.bmm(encoder_outputs)

            # combined_decoder_input: batch x output_size (1) x input_size (l_output_size+hidden_size+4)
            combined_decoder_input = torch.cat((xy_decoder_input, wh_decoder_input, 
                l_decoder_input, context), dim=2)
        else:
            combined_decoder_input = torch.cat((xy_decoder_input, wh_decoder_input, 
                l_decoder_input), dim=2)

        ### 2. get the RNN hidden and output ###
        # output: batch x output_size (1) x hidden_size
        # hidden[0]: output_size (1) x batch x hidden_size
        output, hidden = self.rnn(combined_decoder_input, hidden)

        ### 3. sample the bbox labels ###
        # label_softmax: batch x l_output_size
        label_softmax = nn.Softmax(dim=1)(self.l_out(output.contiguous().view(-1, 
            self.hidden_size))).clamp(1e-5, 1)

        ### 4. sample bbox xy ###
        # xy_hidden: batch x (hidden_size+l_output_size)

        if is_training:
            xy_hidden = torch.cat((hidden[0].squeeze(0), next_l_decoder_input), dim=1)
        else:
            xy_hidden = torch.cat((hidden[0].squeeze(0), label_softmax), dim=1)
        # raw_xy_gmm_param: batch x gmm_comp_num*gmm_param_num
        raw_xy_gmm_param = self.xy_out(xy_hidden)
        # xy: batch x 2
        xy_gmm_param = self.get_gmm_params(batch_size, raw_xy_gmm_param)

        ### 5. sample bbox wh ###
        # wh_hidden: batch x (hidden_size+l_output_size+gmm_comp_num*gmm_param_num)
        sampled_xy, sampled_wh = None, None
        if is_training:
            next_x_decoder_input = next_x_decoder_input.unsqueeze(1)
            next_y_decoder_input = next_y_decoder_input.unsqueeze(1)
            next_xy_decoder_input = self.next_xy_embedding(torch.cat((next_x_decoder_input, 
                y_decoder_input), dim=1))
            next_xy_decoder_input = self.next_xy_input_dropout(next_xy_decoder_input)
            wh_hidden = torch.cat((xy_hidden, next_xy_decoder_input), dim=1)
        else:
            # sampling x and y
            pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy = xy_gmm_param
            next_x_decoder_input, next_y_decoder_input = self.sample_next_state(
                pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy)
            sampled_xy = (next_x_decoder_input, next_y_decoder_input)
            next_x_decoder_input = Variable(torch.FloatTensor([next_x_decoder_input] * self.batch_size))
            next_y_decoder_input = Variable(torch.FloatTensor([next_y_decoder_input] * self.batch_size))
            if torch.cuda.is_available():
                next_x_decoder_input = next_x_decoder_input.cuda()
                next_y_decoder_input = next_y_decoder_input.cuda()
            next_x_decoder_input = next_x_decoder_input.unsqueeze(1)
            next_y_decoder_input = next_y_decoder_input.unsqueeze(1)
            next_xy_decoder_input = self.next_xy_embedding(torch.cat((next_x_decoder_input, 
                y_decoder_input), dim=1))
            next_xy_decoder_input = self.next_xy_input_dropout(next_xy_decoder_input)
            next_xy_decoder_input = self.next_xy_embedding(torch.cat((next_x_decoder_input, 
                y_decoder_input), dim=1))
            wh_hidden = torch.cat((xy_hidden, next_xy_decoder_input), dim=1)
        # raw_wh_gmm_param: batch x gmm_comp_num*gmm_param_num
        raw_wh_gmm_param = self.wh_out(wh_hidden)
        # wh: batch x 2
        wh_gmm_param = self.get_gmm_params(batch_size, raw_wh_gmm_param)

        if not is_training:
            pi_wh, u_w, u_h, sigma_w, sigma_h, rho_wh = wh_gmm_param
            next_w_decoder_input, next_h_decoder_input = self.sample_next_state(pi_wh, u_w, 
                u_h, sigma_w, sigma_h, rho_wh)
            sampled_wh = (next_w_decoder_input, next_h_decoder_input)

        return label_softmax, hidden, attn, xy_gmm_param, wh_gmm_param, sampled_xy, sampled_wh

    def forward(self, encoder_hidden=None, encoder_outputs=None, target_l_variables=None, 
        target_x_variables=None, target_y_variables=None, target_w_variables=None, 
        target_h_variables=None, is_training=0, early_stop_len=None):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        target_l_variables, batch_size, max_length = self._validate_args(target_l_variables, 
            encoder_hidden, encoder_outputs)
        # encoder_hidden[0]: num_directions x batch x hidden_size
        # decoder_hidden[0]: 1 x batch x hidden_size*num_directions
        decoder_hidden = self._init_state(encoder_hidden)

        decoder_l_outputs = []
        xy_gmm_params, wh_gmm_params = [], []
        sequence_labels = []
        sampled_xys = []
        sampled_whs = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_l_output, step_attn):
            decoder_l_outputs.append(step_l_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            labels = step_l_output.topk(1)[1].squeeze(1)
            sequence_labels.append(labels)

            eos_batches = labels.data.eq(self.l_eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_labels)

            return labels

        if is_training:
            l_decoder_input = Variable(torch.LongTensor([self.l_sos_id] * batch_size))
            x_decoder_input = Variable(torch.FloatTensor([self.x_mean] * batch_size))
            y_decoder_input = Variable(torch.FloatTensor([self.y_mean] * batch_size))
            w_decoder_input = Variable(torch.FloatTensor([self.w_mean] * batch_size))
            h_decoder_input = Variable(torch.FloatTensor([self.r_mean] * batch_size))
            next_l_decoder_input = Variable(torch.FloatTensor(batch_size, self.l_output_size).zero_())
            if torch.cuda.is_available():
                l_decoder_input = l_decoder_input.cuda()
                x_decoder_input = x_decoder_input.cuda()
                y_decoder_input = y_decoder_input.cuda()
                w_decoder_input = w_decoder_input.cuda()
                h_decoder_input = h_decoder_input.cuda()
                next_l_decoder_input = next_l_decoder_input.cuda()

            for di in range(max_length):
                next_l_decoder_input[next_l_decoder_input != 0] = 0
                for batch_index in range(batch_size):
                    next_l_decoder_input[batch_index, int(target_l_variables[batch_index,di+1])] = 1
                next_x_decoder_input = target_x_variables[:,di+1]
                next_y_decoder_input = target_y_variables[:,di+1]
                next_w_decoder_input = target_w_variables[:,di+1]
                next_h_decoder_input = target_h_variables[:,di+1]

                step_l_output, decoder_hidden, step_attn, xy_gmm_param, wh_gmm_param, \
                sampled_xy, sampled_wh = self.forward_step(l_decoder_input, x_decoder_input, 
                    y_decoder_input, w_decoder_input, h_decoder_input, decoder_hidden, 
                    encoder_outputs, next_l_decoder_input, next_x_decoder_input, 
                    next_y_decoder_input, is_training=is_training)

                xy_gmm_params.append(xy_gmm_param)
                wh_gmm_params.append(wh_gmm_param)

                labels = decode(di, step_l_output, step_attn)

                l_decoder_input = target_l_variables[:,di+1]
                x_decoder_input = next_x_decoder_input
                y_decoder_input = next_y_decoder_input
                w_decoder_input = next_w_decoder_input
                h_decoder_input = next_h_decoder_input
        else:
            l_decoder_input = Variable(torch.LongTensor([self.l_sos_id] * batch_size))
            x_decoder_input = Variable(torch.FloatTensor([self.x_mean] * batch_size))
            y_decoder_input = Variable(torch.FloatTensor([self.y_mean] * batch_size))
            w_decoder_input = Variable(torch.FloatTensor([self.w_mean] * batch_size))
            h_decoder_input = Variable(torch.FloatTensor([self.r_mean] * batch_size))
            if torch.cuda.is_available():
                l_decoder_input = l_decoder_input.cuda()
                x_decoder_input = x_decoder_input.cuda()
                y_decoder_input = y_decoder_input.cuda()
                w_decoder_input = w_decoder_input.cuda()
                h_decoder_input = h_decoder_input.cuda()

            for di in range(early_stop_len):
                step_l_output, decoder_hidden, step_attn, xy_gmm_param, wh_gmm_param, \
                sampled_xy, sampled_wh = self.forward_step(l_decoder_input, x_decoder_input, 
                    y_decoder_input, w_decoder_input, h_decoder_input, decoder_hidden, 
                    encoder_outputs, is_training=is_training)

                labels = decode(di, step_l_output, step_attn)
                l_decoder_input[0] = labels
                x_decoder_input[0] = sampled_xy[0]
                y_decoder_input[0] = sampled_xy[1]
                w_decoder_input[0] = sampled_wh[0]
                h_decoder_input[0] = sampled_wh[1]

                xy_gmm_params.append(xy_gmm_param)
                wh_gmm_params.append(wh_gmm_param)
                sampled_xys.append(sampled_xy)
                sampled_whs.append(sampled_wh)

                if int(labels.data) == self.l_eos_id:
                    break

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_labels
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()
        ret_dict[DecoderRNN.KEY_XYS] = sampled_xys
        ret_dict[DecoderRNN.KEY_WHS] = sampled_whs

        return decoder_l_outputs, xy_gmm_params, wh_gmm_params, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            inputs = Variable(torch.LongTensor([self.sos_id] * batch_size),
                                    volatile=True).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        assert self.batch_size == batch_size

        return inputs, batch_size, max_length

    def sample_next_state(self, pi, u_x, u_y, sigma_x, sigma_y, rho_xy):
        temperature = 0.4
        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf)/temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = pi.data[0,:].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(self.gmm_comp_num, p=pi)
        # get mixture params:
        u_x = u_x.data[0,pi_idx]
        u_y = u_y.data[0,pi_idx]
        sigma_x = sigma_x.data[0,pi_idx]
        sigma_y = sigma_y.data[0,pi_idx]
        rho_xy = rho_xy.data[0,pi_idx]
        x, y = self.sample_bivariate_normal(u_x, u_y, sigma_x, sigma_y, rho_xy, 
            temperature, greedy=False)
        return x, y

    def get_gmm_params(self, batch_size, gmm_params):
        # parse gmm_params: pi (gmm_comp_num), u_x (gmm_comp_num), u_y (gmm_comp_num), 
        #                   sigma_x (gmm_comp_num), sigma_y (gmm_comp_num), rho_xy (gmm_comp_num)
        # pi: batch x gmm_comp_num
        pi, u_x, u_y, sigma_x, sigma_y, rho_xy = torch.split(gmm_params, self.gmm_comp_num, dim=1)

        pi = nn.Softmax(dim=1)(pi)
        sigma_x = torch.exp(sigma_x)
        sigma_y = torch.exp(sigma_y)
        rho_xy = torch.tanh(rho_xy)

        return (pi, u_x, u_y, sigma_x, sigma_y, rho_xy)


    def repackage_hidden(self, hidden):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(hidden) == Variable:
            hidden = Variable(hidden.data)
            if torch.cuda.is_available():
                hidden = hidden.cuda()
            return hidden
        else:
            return tuple(self.repackage_hidden(v) for v in hidden)
			
			
    def sample_bivariate_normal(self, u_x, u_y, sigma_x, sigma_y, rho_xy, 
        temperature, greedy=False):
        # inputs must be floats
        if greedy:
            return u_x, u_y
        mean = [u_x, u_y]
        sigma_x *= np.sqrt(temperature)
        sigma_y *= np.sqrt(temperature)
        cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],\
            [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]
