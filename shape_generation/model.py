import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from miscc.config import cfg
import sys

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

def conv3x3(in_planes, out_planes, padding=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=padding, bias=False)

# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes, norm=nn.BatchNorm2d):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        norm(out_planes * 2),
        GLU())
    return block

# Downsale the spatial size by a factor of 2
def downBlock_3x3(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def downBlock_4x4(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
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
            prev_hidden = Variable(torch.zeros(state_size))
            prev_cell = Variable(torch.zeros(state_size))
            if cfg.CUDA:
                prev_hidden = prev_hidden.cuda()
                prev_cell = prev_cell.cuda()

            prev_state = (prev_hidden, prev_cell)

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

# ############## G networks ###################
class GET_IMAGE_G(nn.Module):
    def __init__(self, nbf):
        super(GET_IMAGE_G, self).__init__()
        self.img = nn.Sequential(
            conv1x1(nbf, 1),
            nn.Sigmoid()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img

class G_NET(nn.Module):
    def __init__(self, num_classes):
        super(G_NET, self).__init__()
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
        self.img_net = GET_IMAGE_G(self.nbf)
        
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
class D_GET_LOGITS(nn.Module):
    def __init__(self, nbf):
        super(D_GET_LOGITS, self).__init__()
        self.outlogits = nn.Sequential(
            nn.Conv2d(nbf//16, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, h_code):
        output = self.outlogits(h_code)
        return output

class INS_D_NET(nn.Module):
    def __init__(self, num_classes):
        super(INS_D_NET, self).__init__()
        self.nbf = num_classes
        self.downsample1 = downBlock_4x4(self.nbf+1, self.nbf//2)
        self.downsample2 = downBlock_4x4(self.nbf//2, self.nbf//4)
        self.downsample3 = downBlock_4x4(self.nbf//4, self.nbf//8)
        self.downsample4 = downBlock_4x4(self.nbf//8, self.nbf//16)
        self.get_logits = D_GET_LOGITS(self.nbf)

    def forward(self, x_var):
        x_code32 = self.downsample1(x_var)
        x_code16 = self.downsample2(x_code32)
        x_code8 = self.downsample3(x_code16)
        x_code4 = self.downsample4(x_code8)
        return x_code4

class GLB_D_NET(nn.Module):
    def __init__(self, num_classes):
        super(GLB_D_NET, self).__init__()
        self.nbf = num_classes
        self.downsample1 = downBlock_4x4(self.nbf+1, self.nbf//2)
        self.downsample2 = downBlock_4x4(self.nbf//2, self.nbf//4)
        self.downsample3 = downBlock_4x4(self.nbf//4, self.nbf//8)
        self.downsample4 = downBlock_4x4(self.nbf//8, self.nbf//16)
        self.get_logits = D_GET_LOGITS(self.nbf)

    def forward(self, x_var):
        x_code32 = self.downsample1(x_var)
        x_code16 = self.downsample2(x_code32)
        x_code8 = self.downsample3(x_code16)
        x_code4 = self.downsample4(x_code8)
        return x_code4


# ############## Pretrained VGG network ##########################
feature_indices = {2, 5, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 42, 45, 48, 51}
classifier_indices = {1, 4, 6}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        #x = nn.Upsample(size=(224, 224), mode='bilinear')(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        results = []
        for index, model in enumerate(self.features):
            x = model(x)
            if index in feature_indices:
                results.append(x)
        x = x.view(x.size(0), -1)
        for index, model in enumerate(self.classifier):
            x = model(x)
            if index in classifier_indices:
                results.append(x)
        return results

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(vgg_cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in vgg_cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

vgg_cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(vgg_cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model_path = cfg.TRAIN.NET_E.replace('text_encoder100.pth', 'vgg19_bn-c79401a0.pth')
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad = False
    return model