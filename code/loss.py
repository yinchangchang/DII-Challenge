# coding=utf8
#########################################################################
# File Name: classify_loss.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月11日 星期二 14时33分59秒
#########################################################################

import sys
sys.path.append('classification/')

# torch
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.regress_loss = nn.SmoothL1Loss()

    def forward(self, font_output, font_target, use_hard_mining=False):
        batch_size = font_output.size(0)
        # font_output = font_output.cpu()
        # font_target = font_target.cpu()



        # font_target = font_target.unsqueeze(-1).expand(font_output.size()).contiguous()
        font_output = self.sigmoid(font_output)
        # font_loss = self.classify_loss(font_output, font_target)
        # return [font_loss, font_loss, font_loss]


        font_output = font_output.view(-1)
        font_target = font_target.view(-1)
        pos_index = font_target == 1
        neg_index = font_target == 0

        assert font_output.size() == font_target.size()
        assert pos_index.size() == font_target.size()
        assert neg_index.size() == font_target.size()

        # print font_output.size(), font_target.size()


        # pos
        # print pos_index.dtype
        # print pos_index.size()
        # print pos_index
        pos_target = font_target[pos_index]
        pos_output = font_output[pos_index]
        # pos_output = font_output.cpu()[pos_index.cpu()].cuda()
        # pos_target = font_target.cpu()[pos_index.cpu()].cuda()
        if use_hard_mining:
            num_hard_pos = max(2, int(0.2 * batch_size))
            if len(pos_output) > num_hard_pos:
                pos_output, pos_target = hard_mining(pos_output, pos_target, num_hard_pos, largest=False, start=int(num_hard_pos/4))
        if len(pos_output):
            pos_loss = self.classify_loss(pos_output, pos_target) * 0.5
        else:
            pos_loss = 0


        # neg
        neg_output = font_output[neg_index]
        neg_target = font_target[neg_index]
        if use_hard_mining:
            num_hard_neg = max(num_hard_pos, 2)
            if len(neg_output) > num_hard_neg:
                neg_output, neg_target = hard_mining(neg_output, neg_target, num_hard_neg, largest=True, start=int(num_hard_pos/4))
        if len(neg_output):
            neg_loss = self.classify_loss(neg_output, neg_target) * 0.5
        else:
            neg_loss = 0

        font_loss = pos_loss + neg_loss
        return [font_loss, pos_loss, neg_loss]
        # return [font_loss.cuda(), pos_loss, neg_loss]


def hard_mining(neg_output, neg_labels, num_hard, largest=True, start=0):
    # num_hard = min(max(num_hard, 10), len(neg_output))
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)), largest=largest)
    start = 0
    idcs = idcs[start:]
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels
