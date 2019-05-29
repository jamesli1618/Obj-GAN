from __future__ import print_function
import math
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch
import sys

class Loss(object):
    """ Base class for encapsulation of the loss functions.

    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.

    Attributes:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.  Implementation depends on individual
            sub-classes.
        acc_loss (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the loss of multiple batches.  Implementation depends on individual
            sub-classes.
    """

    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        '''if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")'''
        # accumulated loss
        self.acc_loss = 0
        # normalization term
        self.norm_term = 0

    def reset(self):
        """ Reset the accumulated loss. """
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """ Get the loss.

        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.

        Returns:
            loss (float): value of the loss.
        """
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        """ Evaluate and accumulate loss given outputs and expected results.

        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.  Override it to define your own accumulation
        method.

        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError

    def cuda(self):
        if self.criterion is not None:
            self.criterion.cuda()

    def backward(self):
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward()

class NLLLoss(Loss):
    """ Batch averaged negative log-likelihood loss.

    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
    """

    _NAME = "Avg NLLLoss"

    def __init__(self, weight=None, mask=None, size_average=True):
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask.")
            weight[mask] = 0

        super(NLLLoss, self).__init__(
            self._NAME,
            nn.NLLLoss(weight=weight, size_average=size_average))

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.data[0]
        if self.size_average:
            # average loss per batch
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1

class Perplexity(NLLLoss):
    """ Language model perplexity loss.

    Perplexity is the token averaged likelihood.  When the averaging options are the
    same, it is the exponential of negative log-likelihood.

    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
    """

    _NAME = "Perplexity"
    _MAX_EXP = 100

    def __init__(self, weight=None, mask=None, lamda=1.0):
        self.lamda = lamda
        super(Perplexity, self).__init__(weight=weight, mask=mask, size_average=False)

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        if self.mask is None:
            self.norm_term += np.prod(target.size())
        else:
            self.norm_term += target.data.ne(self.mask).sum()

    def get_loss(self):
        return self.lamda*self.acc_loss/self.norm_term.float()

class BBLoss(Loss):

    _NAME = "Bbox Loss"
    def __init__(self, batch_size, gmm_comp_num, lamda):
        self.batch_size = batch_size
        self.gmm_comp_num = gmm_comp_num
        self.lamda = lamda

        super(BBLoss, self).__init__(self._NAME, criterion=None)

    def eval_batch(self, xy_gmm_params, wh_gmm_params, gt_x, gt_y, gt_w, gt_h, gt_l):
        # xy_gmm_params: (pi, u_x, u_y, sigma_x, sigma_y, rho_xy)
        # wh_gmm_params: (pi, u_x, u_y, sigma_x, sigma_y, rho_xy)
        # pi: batch x gmm_comp_num
        # u_x: batch x gmm_comp_num
        # u_y: batch x gmm_comp_num
        # sigma_x: batch x gmm_comp_num
        # sigma_y: batch x gmm_comp_num
        # rho_xy: batch x gmm_comp_num
        # gt_x: batch

        # 1. get gmms
        pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy = xy_gmm_params
        pi_wh, u_w, u_h, sigma_w, sigma_h, rho_wh = wh_gmm_params

        batch_size, gmm_comp_num = pi_xy.size()
        pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy = pi_xy.contiguous().view(-1), \
            u_x.contiguous().view(-1), u_y.contiguous().view(-1), sigma_x.contiguous().view(-1), \
            sigma_y.contiguous().view(-1), rho_xy.contiguous().view(-1)
        pi_wh, u_w, u_h, sigma_w, sigma_h, rho_wh = pi_wh.contiguous().view(-1), \
            u_w.contiguous().view(-1), u_h.contiguous().view(-1), sigma_w.contiguous().view(-1), \
            sigma_h.contiguous().view(-1), rho_wh.contiguous().view(-1)

        # 3. calculate the bbox loss
        mask = (gt_l != 0).float()
        gt_x = gt_x.unsqueeze(1).repeat(1, gmm_comp_num).view(-1)
        gt_y = gt_y.unsqueeze(1).repeat(1, gmm_comp_num).view(-1)
        gt_w = gt_w.unsqueeze(1).repeat(1, gmm_comp_num).view(-1)
        gt_h = gt_h.unsqueeze(1).repeat(1, gmm_comp_num).view(-1)

        xy_pdf = self.pdf(pi_xy, gt_x, gt_y, u_x, u_y, sigma_x, sigma_y, rho_xy, batch_size, gmm_comp_num)
        wh_pdf = self.pdf(pi_wh, gt_w, gt_h, u_w, u_h, sigma_w, sigma_h, rho_wh, batch_size, gmm_comp_num)
        bbox_loss = (-torch.sum(mask*xy_pdf)-torch.sum(mask*wh_pdf))#/(gmm_comp_num*batch_size)
        #print('bbox_loss: ', bbox_loss)

        self.acc_loss += bbox_loss
        self.norm_term += torch.sum(mask)

    def get_loss(self):
        return self.lamda*self.acc_loss/self.norm_term

    def pdf(self, pi_xy, x, y, u_x, u_y, sigma_x, sigma_y, rho_xy, batch_size, gmm_comp_num):
        # all inputs have the same shape: batch*gmm_comp_num
        z_x = ((x-u_x)/sigma_x)**2
        z_y = ((y-u_y)/sigma_y)**2
        z_xy = (x-u_x)*(y-u_y)/(sigma_x*sigma_y)
        z = z_x + z_y - 2*rho_xy*z_xy
        a = -z/(2*(1-rho_xy**2))
        a = a.view(batch_size, gmm_comp_num)
        a_max = torch.max(a, dim=1)[0]
        a_max = a_max.unsqueeze(1).repeat(1, gmm_comp_num)
        a, a_max = a.view(-1), a_max.view(-1)

        exp = torch.exp(a-a_max)
        norm = torch.clamp(2*np.pi*sigma_x*sigma_y*torch.sqrt(1-rho_xy**2), min=1e-5)
        raw_pdf = pi_xy*exp/norm
        raw_pdf = raw_pdf.view(batch_size, gmm_comp_num)
        raw_pdf = torch.log(torch.sum(raw_pdf, dim=1)+1e-5)
        a_max = a_max.view(batch_size, gmm_comp_num)[:,0]
        raw_pdf = raw_pdf+a_max

        return raw_pdf
