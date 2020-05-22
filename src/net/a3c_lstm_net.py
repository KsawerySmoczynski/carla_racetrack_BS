from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils import norm_col_init, weights_init

# https://github.com/dgriff777/rl_a3c_pytorch/blob/master/model.py

class A3C_LSTM(torch.nn.Module):
    def __init__(self, n_img_inputs, n_numeric_imputs, rgb:bool=False):
        #Num inputs to szerokość
        super(A3C_LSTM, self).__init__()
        self.rgb = rgb
        self.conv_depth = nn.Conv2d(n_img_inputs, 64, 5, stride=1, padding=2)
        self.conv_depth2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        if self.rgb:
            self.conv_rgb = nn.Conv2d(n_img_inputs, 64, 5, stride=1, padding=2)
            self.conv_rgb2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.maxp2 = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(n_numeric_imputs, 64)

        if self.rgb:
            self.lstm = nn.LSTMCell(512, 256)
            self.critic_linear = nn.Linear(512 + 64, 1)
            self.actor_linear = nn.Linear(512 + 64, 2)
        else:
            self.lstm = nn.LSTMCell(256, 256)
            self.critic_linear = nn.Linear(256 + 64, 1)
            self.actor_linear = nn.Linear(256 + 64, 2)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv_depth.weight.data.mul_(relu_gain)
        self.conv_depth2.weight.data.mul_(relu_gain)
        if self.rgb:
            self.conv_rgb.weight.data.mul_(relu_gain)
            self.conv_rgb2.weight.data.mul_(relu_gain)


        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x_numeric, depth, rgb, (hx, cx) = inputs #-> transform inputs

        x_numeric = self.fc(x_numeric)

        x_depth = F.relu(self.maxp1(self.conv_depth(depth)))
        x = F.relu(self.maxp2(self.conv_depth2(x_depth)))

        if self.rgb:
            x_rgb = F.relu(self.maxp1(self.conv_depth(rgb)))
            x_rgb = F.relu(self.maxp2(self.conv2(x_rgb)))
            x_rgb = x_rgb.view(x_rgb.size(0), -1)
            x = torch.cat((x, x_rgb), dim=1)

        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = torch.cat((x_numeric,x), dim=0)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


