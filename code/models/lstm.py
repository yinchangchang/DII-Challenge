#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *

import numpy as np

import sys
sys.path.append('tools')
import parse, py_op
args = parse.args


def time_encoding_data(d = 512, time = 200):
    vec = np.array([np.arange(time) * i for i in range(d/2)], dtype=np.float32).transpose()
    vec = vec / vec.max() / 2
    encoding = np.concatenate((np.sin(vec), np.cos(vec)), 1)
    encoding = torch.from_numpy(encoding)
    return encoding


class LSTM(nn.Module):
    def __init__(self, opt):
        super ( LSTM, self ).__init__ ( )
        self.use_cat = args.use_cat
        self.avg_time = args.avg_time

        self.embedding = nn.Embedding (opt.vocab_size, opt.embed_size )
        self.lstm = nn.LSTM ( input_size=opt.embed_size,
                              hidden_size=opt.hidden_size,
                              num_layers=opt.num_layers,
                              batch_first=True,
                              bidirectional=True)

        self.linear_embed = nn.Sequential (
            nn.Linear ( opt.embed_size, opt.embed_size ),
            nn.ReLU ( ),
            nn.Linear ( opt.embed_size, opt.embed_size ),
        )
        self.tv_mapping = nn.Sequential (
            nn.Linear ( opt.embed_size , opt.embed_size / 2),
            nn.ReLU ( ),
            nn.Dropout ( 0.25 ),
            nn.Linear ( opt.embed_size / 2, opt.embed_size ),
        )
        self.alpha = nn.Linear(args.embed_size, 1)


        no = 1
        if self.use_cat:
            no += 1
        self.output_time = nn.Sequential (
                nn.Linear(opt.embed_size * no, opt.embed_size),
                nn.ReLU ( ),
        )

        time = 200
        self.time_encoding = nn.Embedding.from_pretrained(time_encoding_data(opt.embed_size, time))
        self.time_mapping = nn.Sequential (
            nn.Linear ( opt.embed_size, opt.embed_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.25 ),
            nn.Linear ( opt.embed_size, opt.embed_size)
            )

        self.embed_linear = nn.Sequential (
            nn.Linear ( opt.embed_size, opt.embed_size),
            nn.ReLU ( ),
            # nn.Dropout ( 0.25 ),
            # nn.Linear ( opt.embed_size, opt.embed_size),
            # nn.ReLU ( ),
            nn.Dropout ( 0.25 ),
        )
        self.relu = nn.ReLU ( )

        self.linears = nn.Sequential (
            nn.Linear ( opt.hidden_size * 2, opt.rnn_size ),
            # nn.ReLU ( ),
            # nn.Dropout ( 0.25 ),
            # nn.Linear ( opt.rnn_size, opt.rnn_size ),
            nn.ReLU ( ),
            nn.Dropout ( 0.25 ),
            nn.Linear ( opt.rnn_size, 1),
        )
        mn = 128
        self.master_linear = nn.Sequential (
            nn.Linear ( 43, mn),
            # nn.ReLU ( ),
            # nn.Dropout ( 0.25 ),
            # nn.Linear ( mn, mn),
            nn.ReLU ( ),
            nn.Dropout ( 0.25 ),
            nn.Linear ( mn, 1),
        )
        self.output = nn.Sequential (
            nn.Linear ( mn + opt.rnn_size , opt.rnn_size),
            nn.ReLU ( ),
            nn.Linear ( opt.rnn_size, mn),
            nn.ReLU ( ),
            nn.Dropout ( 0.25 ),
            nn.Linear ( mn, 1),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.opt = opt

    def visit_pooling(self, x, mask, time, value=None, trend=None):



        output = x
        size = output.size()
        output = output.view(size[0] * size[1], size[2], output.size(3)) # (bs*98, 72, 512)
        if args.use_glp:
            output = torch.transpose(output, 1,2).contiguous() # (bs*98, 512, 72)
            output = self.pooling(output)
        else:
            weight = self.alpha(output) # (bs*98, 72, 1)
            # print weight.size()
            weight = weight.view(size[0]*size[1], size[2])
            # print weight.size()
            weight = F.softmax(weight)
            x = weight.data.cpu().numpy()
            # print x.shape
            weight = weight.view(size[0]*size[1], size[2], 1).expand(output.size())
            output = weight * output # (bs*98, 512, 72)
            # print output.size()
            output = output.sum(1)
            # print output.size()
            # output = torch.transpose(output, 1,2).contiguous() 
        output = output.view(size[0], size[1], size[3])

        # time encoding
        time = - time.long()
        time = self.time_encoding(time)
        time = self.time_mapping(time)

        if self.use_cat:
            output = torch.cat((output, time), 2) 
            output = self.relu(output)
            output = self.output_time(output) 
        else:
            output = output + time
            output = self.relu(output)



        return output


    def forward_2(self, x, master, mask=None, time=None, phase='train', value=None, trend=None):
        '''
        task2
        '''
        size = list(x.size())
        x = x.view(-1)
        x = self.embedding( x )
        x = self.embed_linear(x)
        size.append(-1)
        x = x.view(size)
        if mask is not None:
            x = self.visit_pooling(x, mask, time, value, trend)
        lstm_out, _ = self.lstm( x )
        lstm_out = torch.transpose(lstm_out, 1, 2).contiguous() # (bs, 512, 98)
        mask = self.pooling(mask)
        # print 'mask', mask.size()
        pool_out = []
        mask_out = []
        time_out = []
        time = time.data.cpu().numpy()
        if phase == 'train':
            start, delta = 4, 6
        else:
            start, delta = 1, 1
        for i in range(start, lstm_out.size(2), delta):
            pool_out.append(self.pooling(lstm_out[:,:, :i]))
            mask_out.append(mask[:, i])
            time_out.append(time[:, i])
        pool_out.append(self.pooling(lstm_out))
        mask_out.append(mask[:, 0])
        time_out.append(np.zeros(size[0]) - 4)

        lstm_out = torch.cat(pool_out, 2)  # (bs, 512, 98)
        mask_out = torch.cat(mask_out, 1)  # (bs, 98)
        time_out = np.array(time_out).transpose() # (bs, 98)

        # print 'lstm_out', lstm_out.size()
        # print 'mask_out', mask_out.size()
        # print err

        lstm_out = torch.transpose(lstm_out, 1, 2).contiguous() # (bs, 98, 512)

        out_vital = self.linears(lstm_out)
        size = list(out_vital.size())
        out_vital = out_vital.view(size[:2])
        out_master = self.master_linear(master).expand(size[:2])
        out = out_vital + out_master
        return out, mask_out, time_out

    def forward_1(self, x, master, mask=None, time=None, phase='train', value=None, trend=None):
        # out = self.master_linear(master)
        size = list(x.size())
        x = x.view(-1)
        x = self.embedding( x )
        # print x.size()
        x = self.embed_linear(x)
        size.append(-1)
        x = x.view(size)
        if mask is not None:
            x = self.visit_pooling(x, mask, time, value, trend)
        lstm_out, _ = self.lstm( x )

        lstm_out = torch.transpose(lstm_out, 1, 2).contiguous()
        lstm_out = self.pooling(lstm_out)
        lstm_out = lstm_out.view(lstm_out.size(0), -1)

        out = self.linears(lstm_out) + self.master_linear(master)
        return out

    def forward(self, x, master, mask=None, time=None, phase='train', value=None, trend=None):
        if args.task == 'task2':
            return self.forward_2(x, master, mask, time, phase, value, trend)
            # return self.forward_1(x, master, mask, time, phase, value, trend)
        else:
            return self.forward_1(x, master, mask, time, phase, value, trend)



