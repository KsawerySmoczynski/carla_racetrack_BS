from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils import norm_col_init, weights_init

# https://github.com/dgriff777/rl_a3c_pytorch/blob/master/model.py
# torch.nn only supports mini-batches The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.
# For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
# If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.


class DDPG(torch.nn.Module):
    def __init__(self, depth_shape, numeric_shape, rgb_shape=None, rgb:bool=False, linear_hidden:int=256, conv_hidden:int=32):
        '''

        :param depth_shape: no channels
        :param rgb_shape: no channels
        :param n_numeric_inputs:
        :param rgb:
        '''
        assert (conv_hidden % 2 == 0), 'conv hidden has to be even number'
        # assert (conv_hidden % 4 == 0), 'conv hidden has to be bigger than 4 number'

        #Num inputs to szerokość
        super(DDPG, self).__init__()
        self.conv_depth = nn.Conv2d(depth_shape[0], conv_hidden, 5, stride=1, padding=2)
        self.conv_depth2 = nn.Conv2d(conv_hidden, int(conv_hidden*2), 3, stride=1, padding=1)
        self.conv_depth3 = nn.Conv2d(int(conv_hidden*2), int(conv_hidden), 2, stride=1, padding=1)
        self.conv_depth4 = nn.Conv2d(int(conv_hidden), int(conv_hidden), 1, stride=1, padding=1)
        # if list(rgb_shape):
        #     self.conv_rgb = nn.Conv2d(rgb_shape[0], 64, 5, stride=1, padding=2)
        #     self.conv_rgb2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        #may be a problem, just add maxpool to particular layers
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.maxp4 = nn.MaxPool2d(2, 2)

        conv_out_size = self._get_conv_out(depth_shape)

        self.linear = nn.Linear(numeric_shape[0], linear_hidden)
        self.linear_conv = nn.Linear(conv_out_size, linear_hidden)
        self.linear2 = nn.Linear(2*linear_hidden, linear_hidden)

        # if list(rgb_shape):
        #     fc2_input += self.conv_rgb2.out_channels
        #     self.linear2 = nn.Linear(fc2_input, 256)
        # else:



        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv_depth.weight.data.mul_(relu_gain)
        self.conv_depth2.weight.data.mul_(relu_gain)
        self.conv_depth3.weight.data.mul_(relu_gain)
        self.conv_depth4.weight.data.mul_(relu_gain)
        # if list(rgb_shape):
        #     self.conv_rgb.weight.data.mul_(relu_gain)
        #     self.conv_rgb2.weight.data.mul_(relu_gain)

        self.linear.weight.data = norm_col_init(self.linear.weight.data, 1.0)
        self.linear.bias.data.fill_(0)
        self.linear_conv.weight.data = norm_col_init(self.linear_conv.weight.data, 1.0)
        self.linear_conv.bias.data.fill_(0)
        self.linear2.weight.data = norm_col_init(self.linear2.weight.data, 1.0)
        self.linear2.bias.data.fill_(0)

        self.train()

    #Change to property
    def name(self):
        return f'{self.__class__.__name__}_l{self.linear.out_features}_conv{self.conv_depth.out_channels}'

    def forward(self, inputs):
        x_numeric, depth, rgb = inputs #-> transform inputs

        x_numeric = self.linear(x_numeric)

        #Adhoc conversion from rgb to depth
        x = F.relu(self.maxp1(self.conv_depth(depth)))
        x = F.relu(self.maxp2(self.conv_depth2(x)))
        x = F.relu(self.maxp3(self.conv_depth3(x)))
        x = F.relu(self.maxp4(self.conv_depth4(x)))


        # if self.rgb:
        #     x_rgb = F.relu(self.maxp1(self.conv_rgb(rgb)))
        #     x_rgb = F.relu(self.maxp2(self.conv_rgb2(x_rgb)))
        #     # x_rgb = x_rgb.view(x_rgb.size(0), -1)
        #     x = torch.cat((x, x_rgb), dim=1)
        x = x.view(x.size(0), -1)
        x = self.linear_conv(x)
        x = torch.cat((x_numeric, x), dim=1)
        x = self.linear2(x)

        return x

    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        x = F.relu(self.maxp1(self.conv_depth(x)))
        x = F.relu(self.maxp2(self.conv_depth2(x)))
        x = F.relu(self.maxp3(self.conv_depth3(x)))
        x = F.relu(self.maxp4(self.conv_depth4(x)))
        return int(np.prod(x.size()))


class DDPGActor(DDPG):
    def __init__(self, depth_shape, numeric_shape, output_shape, rgb:bool=False, rgb_shape:bool=None, cuda:bool=True, linear_hidden:int=256, conv_hidden:int=32):
        super(DDPGActor, self).__init__(depth_shape=depth_shape, numeric_shape=numeric_shape, rgb_shape=rgb_shape,
                                        rgb=rgb, linear_hidden=linear_hidden, conv_hidden=conv_hidden)
        self.linear3 = nn.Linear(self.linear2.out_features, int(linear_hidden/2))
        self.actor_linear = nn.Linear(self.linear3.out_features, output_shape[0])

        self.linear3.weight.data = norm_col_init(self.linear3.weight.data, 1.0)
        self.linear3.bias.data.fill_(0)
        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        if cuda:
            self.cuda()

    def forward(self, inputs):
        x_numeric, depth, rgb = *inputs, None  # -> transform inputs

        x_numeric = self.linear(x_numeric)

        x = F.relu(self.maxp1(self.conv_depth(depth)))
        x = F.relu(self.maxp2(self.conv_depth2(x)))
        x = F.relu(self.maxp3(self.conv_depth3(x)))
        x = F.relu(self.maxp4(self.conv_depth4(x)))

        # if self.rgb:
        #     x_rgb = F.relu(self.maxp1(self.conv_rgb(rgb)))
        #     x_rgb = F.relu(self.maxp2(self.conv_rgb2(x_rgb)))
        #     # x_rgb = x_rgb.view(x_rgb.size(0), -1)
        #     x = torch.cat((x, x_rgb), dim=1)

        x = x.view(x.size(0), -1)
        x = self.linear_conv(x)
        x = torch.cat((x_numeric, x), dim=1)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.actor_linear(x)
        # x = torch.round(x * 1e3) / 1e3 #rounding to 3 decimal places
        return x


class DDPGCritic(DDPG):
    def __init__(self, actor_out_shape, depth_shape, numeric_shape, rgb_shape:bool=None, rgb:bool=False,
                 cuda:bool=True, linear_hidden:int=256, conv_hidden:int=32):
        super(DDPGCritic, self).__init__(depth_shape=depth_shape, numeric_shape=numeric_shape, rgb_shape=rgb_shape,
                                        rgb=rgb, linear_hidden=linear_hidden, conv_hidden=conv_hidden)

        self.linear_actor = nn.Linear(actor_out_shape[0], linear_hidden)
        self.linear3 = nn.Linear(self.linear_actor.out_features + self.linear2.out_features, int(linear_hidden/2))
        self.critic_linear = nn.Linear(self.linear3.out_features, 1)

        self.linear_actor.weight.data = norm_col_init(self.linear_actor.weight.data, 1.0)
        self.linear_actor.bias.data.fill_(0)
        self.linear3.weight.data = norm_col_init(self.linear3.weight.data, 1.0)
        self.linear3.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        if cuda:
            self.cuda()


    def forward(self, inputs):
        x_actor, x_numeric, depth, rgb  = *inputs, None #-> transform inputs

        x_numeric = self.linear(x_numeric)

        #Adhoc conversion from rgb to depth
        x = F.relu(self.maxp1(self.conv_depth(depth)))
        x = F.relu(self.maxp2(self.conv_depth2(x)))
        x = F.relu(self.maxp3(self.conv_depth3(x)))
        x = F.relu(self.maxp4(self.conv_depth4(x)))

        # if self.rgb:
        #     x_rgb = F.relu(self.maxp1(self.conv_depth(rgb)))
        #     x_rgb = F.relu(self.maxp2(self.conv2(x_rgb)))
        #     x_rgb = x_rgb.view(x_rgb.size(0), -1)
        #     x = torch.cat((x, x_rgb), dim=1)

        x = x.view(x.size(0), -1)
        x = self.linear_conv(x)
        x = torch.cat((x_numeric, x), dim=1)
        x = self.linear2(x)
        x_actor = x_actor.view(x_actor.size(0), -1)
        x_actor = self.linear_actor(x_actor)
        #dodaj warstwę liniową dla akcji aktora
        x = torch.cat((x_actor, x), dim=1)
        x = self.linear3(x)
        #dodaj dodatkową warstwę przed krytykiem
        x = self.critic_linear(x)

        return x