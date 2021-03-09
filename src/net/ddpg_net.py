from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils import norm_col_init, weights_init


class DDPG(torch.nn.Module):
    def __init__(self, img_shape, numeric_shape, linear_hidden: int = 256, conv_filters: int = 64):
        '''

        :param img_shape: no channels
        :param rgb_shape: no channels
        :param n_numeric_inputs:
        :param rgb:
        '''
        assert (conv_filters % 2 == 0), 'conv hidden has to be even number'

        super(DDPG, self).__init__()
        self.img_shape = img_shape
        self.numeric_shape = numeric_shape
        self.linear_hidden = linear_hidden
        self.conv_filters = conv_filters
        self.conv = nn.Conv2d(img_shape[0], conv_filters, 5, stride=4, padding=2)
        self.conv2 = nn.Conv2d(conv_filters, conv_filters*2, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(conv_filters*2, int(conv_filters * 2), 4, stride=3, padding=2)
        self.conv4 = nn.Conv2d(int(conv_filters * 2), int(conv_filters * 4), 4, stride=3, padding=1)
        self.conv5 = nn.Conv2d(int(conv_filters * 4), int(conv_filters * 4), 2, stride=2, padding=1)
        self.conv6 = nn.Conv2d(int(conv_filters * 4), int(conv_filters * 4), 2, stride=2, padding=1)

        conv_out_size = self._get_conv_out(img_shape)

        self.linear = nn.Linear(numeric_shape[0], int(linear_hidden / 2))
        self.linear_conv = nn.Linear(conv_out_size, linear_hidden)
        self.linear2 = nn.Linear(self.linear_conv.out_features + self.linear.out_features,
                                 int(linear_hidden / 2))

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        self.conv5.weight.data.mul_(relu_gain)
        self.conv6.weight.data.mul_(relu_gain)

        self.linear.weight.data = norm_col_init(self.linear.weight.data, 1.0)
        self.linear.bias.data.fill_(0)
        self.linear_conv.weight.data = norm_col_init(self.linear_conv.weight.data, 1.0)
        self.linear_conv.bias.data.fill_(0)
        self.linear2.weight.data = norm_col_init(self.linear2.weight.data, 1.0)
        self.linear2.bias.data.fill_(0)

        self.train()

    @property
    def name(self):
        return f'{self.__class__.__name__}_l{self.linear_hidden}_conv{self.conv_filters}'

    def dict(self):
        info = {'img_shape': self.img_shape,
                'numeric_shape': self.numeric_shape,
                'linear_hidden': self.linear_hidden,
                'conv_filters': self.conv_filters}
        return info

    def forward(self, x_numeric: torch.Tensor, depth: torch.Tensor,
                rgb: torch.Tensor = None, **kwargs) -> object:
        x_numeric = torch.tanh(self.linear(x_numeric))

        # Adhoc conversion from rgb to img
        x = F.relu(self.conv(depth))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = x.view(x.size(0), -1)
        x = torch.tanh(self.linear_conv(x))
        x = torch.cat((x_numeric, x), dim=1)
        x = torch.tanh(self.linear2(x))

        return x

    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        x = F.relu(self.conv(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return int(np.prod(x.size()))


#TODO add dict representation
class DDPGActor(DDPG):
    def __init__(self, img_shape, numeric_shape, output_shape,
                 linear_hidden: int = 256, conv_filters: int = 32, cuda: bool = True):
        super(DDPGActor, self).__init__(img_shape=img_shape, numeric_shape=numeric_shape,
                                        linear_hidden=linear_hidden, conv_filters=conv_filters)

        self.actor_linear = nn.Linear(self.linear2.out_features, output_shape[0])

        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        if cuda:
            self.cuda()


    def forward(self, x_numeric: torch.Tensor, img: torch.Tensor, **kwargs) -> object:
        x_numeric = torch.tanh(self.linear(x_numeric))

        x = F.relu(self.conv(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = x.view(x.size(0), -1)
        x = torch.tanh(self.linear_conv(x))
        x = torch.cat((x_numeric, x), dim=1)
        x = torch.tanh(self.linear2(x))
        x = F.hardtanh(self.actor_linear(x))
        return x


#TODO add dict representation
class DDPGCritic(DDPG):
    def __init__(self, actor_out_shape, img_shape, numeric_shape, linear_hidden: int = 256, conv_filters: int = 32, cuda: bool = True):
        super(DDPGCritic, self).__init__(img_shape=img_shape, numeric_shape=numeric_shape,
                                         linear_hidden=linear_hidden, conv_filters=conv_filters)

        self.linear_actor = nn.Linear(actor_out_shape[0], int(linear_hidden / 4))
        self.critic_linear = nn.Linear(self.linear_actor.out_features + self.linear2.out_features, 1)

        self.linear_actor.weight.data = norm_col_init(self.linear_actor.weight.data, 1.0)
        self.linear_actor.bias.data.fill_(0)

        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        if cuda:
            self.cuda()

    def forward(self, action: torch.Tensor, x_numeric: torch.Tensor,
                img: torch.Tensor, **kwargs) -> object:

        x_numeric = torch.tanh(self.linear(x_numeric))

        x = F.relu(self.conv(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = x.view(x.size(0), -1)
        x = torch.tanh(self.linear_conv(x))
        x = torch.cat((x_numeric, x), dim=1)
        x = torch.tanh(self.linear2(x))

        action = action.view(action.size(0), -1)
        action = torch.tanh(self.linear_actor(action))
        x = torch.cat((action, x), dim=1)
        x = self.critic_linear(x)

        return x
