import torch.nn as nn
import torch.nn.functional as F
import torch
import sys

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_l_variables=None, 
      target_x_variables=None, target_y_variables=None, target_w_variables=None, 
      target_h_variables=None, is_training=0, early_stop_len=None):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        result = self.decoder(encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              target_l_variables=target_l_variables,
                              target_x_variables=target_x_variables,
                              target_y_variables=target_y_variables,
                              target_w_variables=target_w_variables,
                              target_h_variables=target_h_variables,
                              is_training=is_training,
                              early_stop_len=early_stop_len)
        return result
