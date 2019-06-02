import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

from miscc.config import cfg
from miscc.utils import pprocess_bt_attns, _get_rois_blob
from GlobalAttention import GlobalAttentionGeneral as ATT_NET
from GlobalAttention import GlobalBUAttentionGeneral as BT_ATT_NET
from models.roi_align.modules.roi_align import RoIAlignAvg


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes, norm=nn.BatchNorm2d):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        norm(out_planes * 2),
        GLU())
    return block


def downBlock_G(in_planes, out_planes, kernel_size=3, stride=2, padding=1, norm=None):
    sequence = [nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
        stride=stride, padding=padding, bias=False)]
    if norm is not None:
        sequence += [norm(out_planes)]
    sequence += [nn.LeakyReLU(0.2, inplace=True)]
    block = nn.Sequential(*sequence)
        
    return block


class HmapResBlock(nn.Module):
    def __init__(self, channel_num):
        super(HmapResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel_num, channel_num * 2, kernel_size=3, 
                stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channel_num * 2),
            GLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel_num, channel_num, kernel_size=3, 
                stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def post_process_words(self, words_emb, max_len):
        batch_size, cur_len = words_emb.size(0), words_emb.size(2)
        new_words_emb = Variable(torch.zeros(batch_size, self.nhidden*self.num_directions, max_len))
        if cfg.CUDA:
            new_words_emb = new_words_emb.cuda()
        new_words_emb[:,:, :cur_len] = words_emb

        return new_words_emb

    def forward(self, captions, cap_lens, max_len, mask=None):
        batch_size = captions.size(0)
        hidden = self.init_hidden(batch_size)

        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)

        words_emb = self.post_process_words(words_emb, max_len)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        model = models.inception_v3()
        model_path = cfg.TRAIN.NET_E.replace('text_encoder100.pth', 'inception_v3_google-1a9a5a14.pth')
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', model_path)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code


class INCEPTION_V3(nn.Module):
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        model_path = cfg.TRAIN.NET_E.replace('text_encoder100.pth', 'inception_v3_google-1a9a5a14.pth')
        state_dict = \
            torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        x = input * 0.5 + 0.5
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # --> mean = 0, std = 1
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        #
        # --> fixed-size input: batch x 3 x 299 x 299
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        # 299 x 299 x 3
        x = self.model(x)
        x = nn.Softmax(dim=-1)(x)
        return x


class INCEPTION_V3_FID(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        """
        super(INCEPTION_V3_FID, self).__init__()

        self.resize_input = resize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3()
        model_path = cfg.TRAIN.NET_E.replace('text_encoder100.pth', 'inception_v3_google-1a9a5a14.pth')
        state_dict = \
            torch.load(model_path, map_location=lambda storage, loc: storage)
        inception.load_state_dict(state_dict)
        for param in inception.parameters():
            param.requires_grad = False

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in 
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output 
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.upsample(x, size=(299, 299), mode='bilinear')

        x = x.clone()
        # [-1.0, 1.0] --> [0, 1.0]
        x = x * 0.5 + 0.5
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


# ############## G networks ###################
class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf
        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 8 * 8 * 2, bias=False),
            nn.BatchNorm1d(ngf * 8 * 8 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)

    def forward(self, z_code, c_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/4 x 32 x 32
        """
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 8, 8)
        # state size ngf/2 x 16 x 16
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 32 x 32
        out_code = self.upsample2(out_code)

        return out_code


class INIT_STAGE_G_MAIN(nn.Module):
    def __init__(self, ngf, nef, nef2):
        super(INIT_STAGE_G_MAIN, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.ef_dim2 = nef2
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.GLB_R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        nef = self.ef_dim
        nef2 = self.ef_dim2

        self.bt_att = BT_ATT_NET(ngf, nef)
        self.residual = self._make_layer(HmapResBlock, ngf*3+nef2)
        self.upsample = upBlock(ngf*3+nef2, ngf)

    def forward(self, h_code_hmap, h_code1_sent, c_code, word_embs,
        glove_word_embs, slabels_feat, mask, rois, num_rois, bt_mask, 
        glb_max_num_roi):
        idf, ih, iw = h_code_hmap.size(1), h_code_hmap.size(2), h_code_hmap.size(3)
        
        num_rois = num_rois.data.cpu().numpy().tolist()
        max_num_roi = np.amax(num_rois)
        slabels_feat = slabels_feat[:, :, :max_num_roi]

        if max_num_roi > 0:
            ### compute bottom-up attn and context vectors
            self.bt_att.applyMask(mask)
            # bt_c_code (variable): batch_size x idf x max_num_rois x 1
            # bt_att (variable): batch_size x cap_len x max_num_rois x 1
            bt_c_code, bt_att = self.bt_att(slabels_feat, glove_word_embs, word_embs)
            # bt_mask: batch x max_num_rois x ih x iw -> batch x max_num_rois x num x ih x iw
            bt_mask = bt_mask[:,:max_num_roi]

            bt_code_mask = bt_mask.unsqueeze(2).repeat(1, 1, bt_c_code.size(1), 1, 1)
            bt_c_code = pprocess_bt_attns(bt_c_code, ih, iw, bt_code_mask)

            bt_att_mask = bt_mask.unsqueeze(2).repeat(1, 1, bt_att.size(1), 1, 1)
            bt_att = pprocess_bt_attns(bt_att, ih, iw, bt_att_mask)

            bt_slabels_mask = bt_mask.unsqueeze(2).repeat(1, 1, self.ef_dim2, 1, 1)
            bt_slabels_code = pprocess_bt_attns(slabels_feat, ih, iw, bt_slabels_mask)
        else:
            bt_c_code = Variable(torch.Tensor(c_code.size()).zero_())
            bt_att = Variable(torch.Tensor(att.size()).zero_())
            bt_slabels_code = Variable(torch.Tensor(c_code.size(0), self.ef_dim2, 
                c_code.size(2), c_code.size(3)).zero_())
            if cfg.CUDA:
                bt_c_code = bt_c_code.cuda()
                bt_att = bt_att.cuda()
                bt_slabels_code = bt_slabels_code.cuda()

        out_code = torch.cat((h_code_hmap, h_code1_sent, bt_c_code, bt_slabels_code), 1)
        # state size ngf x 32 x 32
        out_code = self.residual(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample(out_code)

        return out_code


class G_HMAP(nn.Module):
    def __init__(self, ngf, ncf):
        super(G_HMAP, self).__init__()
        self.gf_dim = ngf # the base number of channels: e.g., 24
        self.in_dim = ncf # number of object classes of interest
        self.define_module()

    def define_module(self):
        ncf, ngf = self.in_dim, self.gf_dim
        # Convolution-InstanceNorm-ReLU
        self.conv3x3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ncf, ngf, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True))

        self.downsample1 = downBlock_G(ngf, ngf*2) # output 48 channels

    def forward(self, hmap):
        """
        :param hmap: batch x ncf x hmap_size x hmap_size
        :return: batch x ngf*2 x hmap_size//2 x hmap_size//2
        """
        # state size ngf x hmap_size x hmap_size
        out_code = self.conv3x3(hmap)
        # state size ngf*2 x hmap_size//2 x hmap_size//2
        out_code = self.downsample1(out_code)

        return out_code


class NEXT_STAGE_G_MAIN(nn.Module):
    def __init__(self, ngf, nef, nef2):
        super(NEXT_STAGE_G_MAIN, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.ef_dim2 = nef2
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.LOCAL_R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        nef = self.ef_dim
        nef2 = self.ef_dim2

        self.att = ATT_NET(ngf, nef)
        self.bt_att = BT_ATT_NET(ngf, nef)
        self.residual = self._make_layer(HmapResBlock, ngf*3+nef2)
        self.upsample = upBlock(ngf*3+nef2, ngf)

    def forward(self, h_code, h_code_hmap, c_code, word_embs,
        glove_word_embs, slabels_feat, mask, rois, num_rois, bt_mask,
        glb_max_num_roi):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            glove_word_embs: batch x cdf2 x sourceL (sourceL=seq_len)
            slabels_feat: batch x cdf2 x max_num_roi x 1
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        idf, ih, iw = h_code.size(1), h_code.size(2), h_code.size(3)

        ### compute grid attn and context vectors
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)

        num_rois = num_rois.data.cpu().numpy().tolist()
        max_num_roi = np.amax(num_rois)
        slabels_feat = slabels_feat[:, :, :max_num_roi]

        raw_bt_c_code = Variable(torch.Tensor(c_code.size(0), idf, 
            glb_max_num_roi, 1).zero_())
        if cfg.CUDA:
            raw_bt_c_code = raw_bt_c_code.cuda()

        if max_num_roi > 0:
            ### compute bottom-up attn and context vectors
            self.bt_att.applyMask(mask)
            # bt_c_code (variable): batch_size x idf x max_num_rois x 1
            # bt_att (variable): batch_size x cap_len x max_num_rois x 1
            bt_c_code, bt_att = self.bt_att(slabels_feat, glove_word_embs, word_embs)
            raw_bt_c_code[:,:,:max_num_roi] = bt_c_code
            # bt_mask: batch x max_num_rois x ih x iw -> batch x max_num_rois x num x ih x iw
            bt_mask = bt_mask[:,:max_num_roi]

            bt_code_mask = bt_mask.unsqueeze(2).repeat(1, 1, bt_c_code.size(1), 1, 1)
            bt_c_code = pprocess_bt_attns(bt_c_code, ih, iw, bt_code_mask)

            bt_att_mask = bt_mask.unsqueeze(2).repeat(1, 1, bt_att.size(1), 1, 1)
            bt_att = pprocess_bt_attns(bt_att, ih, iw, bt_att_mask)

            bt_slabels_mask = bt_mask.unsqueeze(2).repeat(1, 1, self.ef_dim2, 1, 1)
            bt_slabels_code = pprocess_bt_attns(slabels_feat, ih, iw, bt_slabels_mask)
        else:
            bt_c_code = Variable(torch.Tensor(c_code.size()).zero_())
            bt_att = Variable(torch.Tensor(att.size()).zero_())
            bt_slabels_code = Variable(torch.Tensor(c_code.size(0), self.ef_dim2, 
                c_code.size(2), c_code.size(3)).zero_())
            if cfg.CUDA:
                bt_c_code = bt_c_code.cuda()
                bt_att = bt_att.cuda()
                bt_slabels_code = bt_slabels_code.cuda()
        
        h_c_code = torch.cat((h_code+h_code_hmap, c_code, bt_c_code, bt_slabels_code), 1)
        out_code = self.residual(h_c_code)

        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)
        raw_bt_c_code = raw_bt_c_code.transpose(1,2).squeeze(-1)

        return out_code, raw_bt_c_code, att, bt_att


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self, num_classes):
        super(G_NET, self).__init__()
        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        nef2 = cfg.TEXT.GLOVE_EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET()
        self.num_classes = num_classes

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1_sent = INIT_STAGE_G(ngf * 4, ncf)
            self.h_net1_hmap = G_HMAP(ngf//2, num_classes)
            self.h_net1_main = INIT_STAGE_G_MAIN(ngf, nef, nef2)
            self.img_net1 = GET_IMAGE_G(ngf)
        # gf x 64 x 64
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2_hmap = G_HMAP(ngf//2, num_classes)
            self.h_net2_main = NEXT_STAGE_G_MAIN(ngf, nef, nef2)
            self.img_net2 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3_hmap = G_HMAP(ngf//2, num_classes)
            self.h_net3_main = NEXT_STAGE_G_MAIN(ngf, nef, nef2)
            self.img_net3 = GET_IMAGE_G(ngf)

    def forward(self, z_code, sent_emb, word_embs, glove_word_embs, 
        slabels_feat, mask, hmaps, rois, fm_rois, num_rois, bt_masks, 
        fm_bt_masks, glb_max_num_roi):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param glove_word_embs: batch x cdf2 x seq_len
            :param slabels_feat: batch x cdf2 x max_num_roi x 1
            :param mask: batch x seq_len
            :return:
        """
        fake_imgs, bt_c_codes, att_maps, bt_att_maps = [], [], [], []
        c_code, mu, logvar = self.ca_net(sent_emb)

        if cfg.TREE.BRANCH_NUM > 0:
            h_code1_hmap = self.h_net1_hmap(hmaps[0])
            h_code1_sent = self.h_net1_sent(z_code, c_code)
            h_code1 = self.h_net1_main(h_code1_hmap, h_code1_sent, 
                c_code, word_embs, glove_word_embs, slabels_feat, 
                mask, fm_rois, num_rois, fm_bt_masks, glb_max_num_roi)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2_hmap = self.h_net2_hmap(hmaps[1])
            h_code2, bt_c_code2, att1, bt_att1 = self.h_net2_main(h_code1, 
                h_code2_hmap, c_code, word_embs, glove_word_embs, slabels_feat, 
                mask, rois[0], num_rois, bt_masks[0], glb_max_num_roi)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            bt_c_codes.append(bt_c_code2)
            if att1 is not None:
                att_maps.append(att1)
            if bt_att1 is not None:
                bt_att_maps.append(bt_att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3_hmap = self.h_net3_hmap(hmaps[2])
            h_code3, bt_c_code3, att2, bt_att2 = self.h_net3_main(h_code2, 
                h_code3_hmap, c_code, word_embs, glove_word_embs, slabels_feat, 
                mask, rois[1], num_rois, bt_masks[1], glb_max_num_roi)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            bt_c_codes.append(bt_c_code3)
            if att2 is not None:
                att_maps.append(att2)
            if bt_att2 is not None:
                bt_att_maps.append(bt_att2)

        return fake_imgs, bt_c_codes, att_maps, bt_att_maps, mu, logvar


# ############## Shape G networks ##########################
# Downsale the spatial size by a factor of 2
def downBlock_3x3(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes, norm=nn.BatchNorm2d):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        norm(out_planes * 2),
        GLU())
    return block


class CLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, input_size, hidden_size, kernel_size, padding):
        super(CLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)).cuda(),
                Variable(torch.zeros(state_size)).cuda()
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ResBlock(nn.Module):
    def __init__(self, channel_num, norm=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            norm(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num * 2),
            norm(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class GET_SHAPE_G(nn.Module):
    def __init__(self, nbf):
        super(GET_SHAPE_G, self).__init__()
        self.img = nn.Sequential(
            conv1x1(nbf, 1),
            nn.Sigmoid()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class SHP_G_NET(nn.Module):
    def __init__(self, num_classes):
        super(SHP_G_NET, self).__init__()
        self.nbf = num_classes
        self.fm_size = cfg.ROI.FM_SIZE

        self.downsample1 = downBlock_3x3(self.nbf, self.nbf*2)
        self.downsample2 = downBlock_3x3(self.nbf*2, self.nbf*4)
        self.fwd_convlstm = CLSTMCell(self.nbf*4, self.nbf*2, 3, 1)
        self.bwd_convlstm = CLSTMCell(self.nbf*4, self.nbf*2, 3, 1)
        self.jointConv = Block3x3_relu(self.nbf*8, self.nbf*4, norm=nn.InstanceNorm2d)
        self.residual = self._make_layer(ResBlock, self.nbf*4)
        self.upsample1 = upBlock(self.nbf*4, self.nbf*2, norm=nn.InstanceNorm2d)
        self.upsample2 = upBlock(self.nbf*2, self.nbf, norm=nn.InstanceNorm2d)
        self.img_net = GET_SHAPE_G(self.nbf)
        
    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num, norm=nn.InstanceNorm2d))
        return nn.Sequential(*layers) 

    def forward(self, z_code, bbox_maps_fwd, bbox_maps_bwd, bbox_fmaps):
        """
            :param z_code: batch x max_num_roi x self.nbf
            :param bbox_maps_fwd: batch x max_num_roi x class_num x hmap_size x hmap_size
            :param bbox_maps_bwd: batch x max_num_roi x class_num x hmap_size x hmap_size
            :param bbox_fmaps: batch x max_num_roi x fmap_size x fmap_size
            :return: 
                   fake_hmaps: batch x max_num_roi x 3 x hmap_size x hmap_size
        """
        batch_size, max_num_roi = z_code.size(0), z_code.size(1)
        # batch x max_num_roi x self.nbf -> batch x max_num_roi x self.nbf x 1
        z_code = z_code.unsqueeze(3)
        # batch x max_num_roi x self.nbf -> batch x max_num_roi x 16 x self.nbf
        z_code = z_code.repeat(1,1,1,self.fm_size**2)
        # batch x max_num_roi x 16 x self.nbf -> batch x max_num_roi x self.nbf x 16 x 16
        z_code = z_code.view(batch_size, max_num_roi, -1, self.fm_size, self.fm_size)

        # 1. downsample
        hmap_size = bbox_maps_fwd.size(3)
        bbox_maps_fwd = bbox_maps_fwd.view(-1, self.nbf, hmap_size, hmap_size)
        h_code_fwd = self.downsample1(bbox_maps_fwd)
        h_code_fwd = self.downsample2(h_code_fwd)
        h_code_fwd = h_code_fwd.view(batch_size, max_num_roi, -1, self.fm_size, self.fm_size)

        bbox_maps_bwd = bbox_maps_bwd.view(-1, self.nbf, hmap_size, hmap_size)
        h_code_bwd = self.downsample1(bbox_maps_bwd)
        h_code_bwd = self.downsample2(h_code_bwd)
        h_code_bwd = h_code_bwd.view(batch_size, max_num_roi, -1, self.fm_size, self.fm_size)

        # 2. conv lstm
        state_fwd, state_bwd = None, None
        state_fwd_lst, state_bwd_lst = [], []
        for t in range(0, max_num_roi):
            state_fwd = self.fwd_convlstm(h_code_fwd[:,t], state_fwd)
            # batch x self.nbf*2 x 16 x 16 -> batch x 1 x self.nbf*2 x 16 x 16
            state_fwd_lst.append(state_fwd[0].unsqueeze(1))
            state_bwd = self.bwd_convlstm(h_code_bwd[:,t], state_bwd)
            state_bwd_lst.append(state_bwd[0].unsqueeze(1))

        # 3. concatenate noise
        h_code = []
        for t in range(0, max_num_roi):
            h_code.append(torch.cat((state_fwd_lst[t], state_bwd_lst[max_num_roi-t-1], z_code[:,t:t+1]), 2))
        h_code = torch.cat(h_code, 1)
        h_code = h_code.view(batch_size*max_num_roi, -1, self.fm_size, self.fm_size)

        # 4. joint conv
        h_code = self.jointConv(h_code)

        # 5. crop
        bbox_fmaps = bbox_fmaps.unsqueeze(2).repeat(1,1,self.nbf*4,1,1).view(batch_size*max_num_roi, -1, 
            self.fm_size, self.fm_size)
        h_code = h_code*bbox_fmaps

        # 6. residual
        h_code = self.residual(h_code)

        # 7. upsample
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)

        # 8. get image
        fake_hmaps = self.img_net(h_code)
        fake_hmaps = fake_hmaps.view(batch_size, max_num_roi, -1, hmap_size, hmap_size)

        return fake_hmaps


# ############## D networks ##########################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_ntimes(ngf, ndf, n_layer):
    sequence =[
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3+ngf, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for n in range(1, n_layer):
        nf_mult_prev = ndf * min(2**(n-1), 8)
        nf_mult = ndf * min(2**n, 8)
        sequence += [
            nn.Conv2d(nf_mult_prev, nf_mult, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

    encode_img = nn.Sequential(*sequence)

    return encode_img


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.layer_num = cfg.GAN.LAYER_D_NUM
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * pow(2, self.layer_num-1) + nef, 
                ndf * pow(2, self.layer_num-1))

        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * pow(2, self.layer_num-1), 1, kernel_size=4, stride=2),
            nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            # conditioning output
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, h_code.size(2), h_code.size(3))
            # state size (ngf+egf) x 8 x 8
            h_c_code = torch.cat((h_code, c_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output

# ############## Patch discriminators ##########################

# For 64 x 64 images
class PAT_D_NET64(nn.Module):
    def __init__(self, b_jcu=True):
        super(PAT_D_NET64, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = cfg.GAN.GF_DIM // 4
        self.img_code = encode_image_by_ntimes(0, ndf, cfg.GAN.LAYER_D_NUM)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code4 = self.img_code(x_var)  # 4 x 4 x 8df (cfg.GAN.LAYER_D_NUM=4)
        return x_code4


# For 128 x 128 images
class PAT_D_NET128(nn.Module):
    def __init__(self, b_jcu=True):
        super(PAT_D_NET128, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = cfg.GAN.GF_DIM // 4
        self.img_code = encode_image_by_ntimes(0, ndf, cfg.GAN.LAYER_D_NUM)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code8 = self.img_code(x_var)  # 8 x 8 x 8df (cfg.GAN.LAYER_D_NUM=4)
        return x_code8


# For 256 x 256 images
class PAT_D_NET256(nn.Module):
    def __init__(self, b_jcu=True):
        super(PAT_D_NET256, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = cfg.GAN.GF_DIM // 4
        self.img_code = encode_image_by_ntimes(0, ndf, cfg.GAN.LAYER_D_NUM)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code16 = self.img_code(x_var)  # 16 x 16 x 8df (cfg.GAN.LAYER_D_NUM=4)
        return x_code16

# ############## Shape discriminators ##########################

# For 64 x 64 images
class SHP_D_NET64(nn.Module):
    def __init__(self, num_classes):
        super(SHP_D_NET64, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = cfg.GAN.GF_DIM // 4
        ncf = num_classes
        self.img_code = encode_image_by_ntimes(ngf, ndf, cfg.GAN.LAYER_D_NUM)
        self.shp_code = nn.Sequential(
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(ncf, ngf, kernel_size=3, padding=0),
                            nn.InstanceNorm2d(ngf),
                            nn.LeakyReLU(0.2, inplace=True))
        self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, x_var, s_var):
        new_s_var = self.shp_code(s_var)
        x_s_var = torch.cat([x_var, new_s_var], dim=1)
        # 64//cfg.GAN.LAYER_D_NUM x 64//cfg.GAN.LAYER_D_NUM x 2^(cfg.GAN.LAYER_D_NUM-1)
        x_code4 = self.img_code(x_s_var)
        return x_code4


# For 128 x 128 images
class SHP_D_NET128(nn.Module):
    def __init__(self, num_classes):
        super(SHP_D_NET128, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = cfg.GAN.GF_DIM // 4
        ncf = num_classes
        self.img_code = encode_image_by_ntimes(ngf, ndf, cfg.GAN.LAYER_D_NUM)
        self.shp_code = nn.Sequential(
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(ncf, ngf, kernel_size=3, padding=0),
                            nn.InstanceNorm2d(ngf),
                            nn.LeakyReLU(0.2, inplace=True))
        self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, x_var, s_var):
        new_s_var = self.shp_code(s_var)
        x_s_var = torch.cat([x_var, new_s_var], dim=1)
        # 128//cfg.GAN.LAYER_D_NUM x 128//cfg.GAN.LAYER_D_NUM x 2^(cfg.GAN.LAYER_D_NUM-1)
        x_code8 = self.img_code(x_s_var)
        return x_code8


# For 256 x 256 images
class SHP_D_NET256(nn.Module):
    def __init__(self, num_classes):
        super(SHP_D_NET256, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = cfg.GAN.GF_DIM // 4
        ncf = num_classes
        self.img_code = encode_image_by_ntimes(ngf, ndf, cfg.GAN.LAYER_D_NUM)
        self.shp_code = nn.Sequential(
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(ncf, ngf, kernel_size=3, padding=0),
                            nn.InstanceNorm2d(ngf),
                            nn.LeakyReLU(0.2, inplace=True))
        self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, x_var, s_var):
        new_s_var = self.shp_code(s_var)
        x_s_var = torch.cat([x_var, new_s_var], dim=1)
        # 256//cfg.GAN.LAYER_D_NUM x 256//cfg.GAN.LAYER_D_NUM x 2^(cfg.GAN.LAYER_D_NUM-1)
        x_code16 = self.img_code(x_s_var)
        return x_code16


# ############## Object discriminators ##########################
# For small-scale
class OBJ_SS_D_NET(nn.Module):
    def __init__(self, num_classes, b_jcu=True):
        super(OBJ_SS_D_NET, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.GLOVE_EMBEDDING_DIM+cfg.GAN.GF_DIM
        ngf = cfg.GAN.GF_DIM // 4
        ncf = num_classes
        self.roi_size = cfg.ROI.ROI_BASE_SIZE
        self.im_scales = np.array([1])
        n_layer = 3

        self.img_code = encode_image_by_ntimes(ngf, ndf, n_layer)
        self.shp_code = nn.Sequential(
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(ncf, ngf, kernel_size=3, padding=0),
                            nn.InstanceNorm2d(ngf),
                            nn.LeakyReLU(0.2, inplace=True))
        self.roi_code = nn.Sequential(
                    nn.Conv2d(ndf * min(2**(n_layer-1), 8), ndf * 4, kernel_size=4, stride=1, padding=1),
                    nn.LeakyReLU(0.2, True))
        self.RoIAlignAvg = RoIAlignAvg(self.roi_size, self.roi_size, 1.0/16.0)

        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf//2, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf//2, nef, bcondition=True)

    def forward(self, x_var, s_var, fm_rois, num_rois, img_size=512):
        fm_rois_roi = fm_rois.data.cpu().numpy()
        fm_rois_roi[:, :, [2, 3]] = fm_rois_roi[:, :, [0, 1]] + fm_rois_roi[:, :, [2, 3]]
        num_rois = num_rois.data.cpu().numpy().tolist()

        x_var = F.interpolate(x_var, size=(img_size, img_size), mode='bilinear', align_corners=True)
        s_var = F.interpolate(s_var, size=(img_size, img_size), mode='bilinear', align_corners=True)
        new_s_var = self.shp_code(s_var)
        x_s_var = torch.cat([x_var, new_s_var], dim=1)
        x_code64 = self.img_code(x_s_var)  # batch x 4df x 64 x 64

        max_num_roi = np.amax(num_rois)
        batch_size = fm_rois_roi.shape[0]
        fm_rois_roi = np.reshape(fm_rois_roi, (fm_rois_roi.shape[0]*fm_rois_roi.shape[1], fm_rois_roi.shape[2]))

        # initilize the tensor holder here.
        roi_data = torch.FloatTensor(1)

        # ship to cuda
        if cfg.CUDA:
            roi_data = roi_data.cuda()
        # make variable
        vroi_data = Variable(roi_data, requires_grad=False)

        # prepare data
        im_scales = np.array([1]*batch_size*cfg.ROI.BOXES_NUM)
        gt_rois_np = _get_rois_blob(fm_rois_roi[:, :4], im_scales)[np.newaxis, :]
        gt_rois_pt = torch.from_numpy(gt_rois_np[np.newaxis, :])
        vroi_data.data.resize_(gt_rois_pt.size()).copy_(gt_rois_pt)
        pooled_feat = self.RoIAlignAvg(x_code64, vroi_data.view(-1,5))
        pooled_feat = self.roi_code(pooled_feat)
        pooled_feat = pooled_feat.view(batch_size, cfg.ROI.BOXES_NUM, 
            pooled_feat.size(1), pooled_feat.size(2), pooled_feat.size(3))

        return pooled_feat


# For large-scale
class OBJ_LS_D_NET(nn.Module):
    def __init__(self, num_classes, b_jcu=True):
        super(OBJ_LS_D_NET, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.GLOVE_EMBEDDING_DIM+cfg.GAN.GF_DIM
        ngf = cfg.GAN.GF_DIM // 4
        ncf = num_classes
        self.roi_size = cfg.ROI.ROI_BASE_SIZE
        self.im_scales = np.array([1])
        n_layer = 4

        self.img_code = encode_image_by_ntimes(ngf, ndf, n_layer)
        self.shp_code = nn.Sequential(
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(ncf, ngf, kernel_size=3, padding=0),
                            nn.InstanceNorm2d(ngf),
                            nn.LeakyReLU(0.2, inplace=True))
        self.roi_code = nn.Sequential(
                    nn.Conv2d(ndf * min(2**(n_layer-1), 8), ndf * 4, kernel_size=4, stride=1, padding=1),
                    nn.LeakyReLU(0.2, True))
        self.RoIAlignAvg = RoIAlignAvg(self.roi_size, self.roi_size, 1.0/16.0)

        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf//2, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf//2, nef, bcondition=True)

    def forward(self, x_var, s_var, fm_rois, num_rois, img_size=512):
        fm_rois_roi = fm_rois.data.cpu().numpy()
        fm_rois_roi[:, :, [2, 3]] = fm_rois_roi[:, :, [0, 1]] + fm_rois_roi[:, :, [2, 3]]
        num_rois = num_rois.data.cpu().numpy().tolist()

        x_var = F.interpolate(x_var, size=(img_size, img_size), mode='bilinear', align_corners=True)
        s_var = F.interpolate(s_var, size=(img_size, img_size), mode='bilinear', align_corners=True)
        new_s_var = self.shp_code(s_var)
        x_s_var = torch.cat([x_var, new_s_var], dim=1)
        x_code32 = self.img_code(x_s_var)  # batch x 4df x 32 x 32

        max_num_roi = np.amax(num_rois)
        batch_size = fm_rois_roi.shape[0]
        fm_rois_roi = np.reshape(fm_rois_roi, (fm_rois_roi.shape[0]*fm_rois_roi.shape[1], fm_rois_roi.shape[2]))

        # initilize the tensor holder here.
        roi_data = torch.FloatTensor(1)

        # ship to cuda
        if cfg.CUDA:
            roi_data = roi_data.cuda()
        # make variable
        vroi_data = Variable(roi_data, requires_grad=False)

        # prepare data
        im_scales = np.array([1]*batch_size*cfg.ROI.BOXES_NUM)
        gt_rois_np = _get_rois_blob(fm_rois_roi[:, :4], im_scales)[np.newaxis, :]
        gt_rois_pt = torch.from_numpy(gt_rois_np[np.newaxis, :])
        vroi_data.data.resize_(gt_rois_pt.size()).copy_(gt_rois_pt)
        pooled_feat = self.RoIAlignAvg(x_code32, vroi_data.view(-1,5))
        pooled_feat = self.roi_code(pooled_feat)
        pooled_feat = pooled_feat.view(batch_size, cfg.ROI.BOXES_NUM, 
            pooled_feat.size(1), pooled_feat.size(2), pooled_feat.size(3))

        return pooled_feat