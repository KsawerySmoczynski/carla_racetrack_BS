#Script for pretraining reference net
import argparse
import os

import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader

from config import NUMERIC_FEATURES, DEVICE, DATE_TIME
from net.ddpg_net import DDPGActor, DDPGCritic

np.random.seed(48)
import pandas as pd
import torch
torch.manual_seed(48)
from torchvision import transforms
from torch import multiprocessing as mp, nn

from net.utils import get_paths, DepthPreprocess, ToSupervised, SimpleDataset, unpack_supervised_batch


#TODO check how pytorch dataloader works and inherit it to build adhoc loading from disk
# https://towardsdatascience.com/deep-learning-model-training-loop-e41055a24b73
# https://towardsdatascience.com/the-false-promise-of-off-policy-reinforcement-learning-algorithms-c56db1b4c79a

# 5.3 Write data to tensorboard and save net if its best epoch


#BASH
#for map in 'circut_spa' 'RaceTrack' 'RaceTrack2'; do for car in 0 1 2; do for speed in 150 110 90 60; do echo $car $speed $map; done; done; done;

#TODO tensorboard
def main(args):

    device = torch.device('cuda:0')

    no_epochs = 5
    batch_size = 128
    depth_shape = [1, 60, 320]

    linear_hidden = 512
    conv_hidden = 64


    #Get train test paths -> later on implement cross val
    steps = get_paths(as_tuples=True, shuffle=True)
    steps_train, steps_test = steps[:int(len(steps)*.8)], steps[int(len(steps)*.2):]

    dataset_train = SimpleDataset(ids=steps_train, batch_size=batch_size, transform=transforms.Compose([DepthPreprocess(), ToSupervised()]))
    dataset_test = SimpleDataset(ids=steps_test, batch_size=batch_size, transform=transforms.Compose([DepthPreprocess(), ToSupervised()]))

    dataloader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': int(mp.cpu_count())} #we've already shuffled paths

    dataset_train = DataLoader(dataset_train, **dataloader_params)
    dataset_test = DataLoader(dataset_test, **dataloader_params)

    #Nets
    actor_net = DDPGActor(depth_shape=depth_shape, numeric_shape=[len(NUMERIC_FEATURES)], output_shape=[2],
                          linear_hidden=linear_hidden, conv_hidden=conv_hidden)
    # critic_net = DDPGCritic(actor_out_shape=[2, ], depth_shape=depth_shape, numeric_shape=[len(NUMERIC_FEATURES)],
    #                         linear_hidden=linear_hidden, conv_hidden=conv_hidden)

    # save path
    path = f'../data/models/{DATE_TIME}/{str(actor_net)}'
    os.makedirs(path, exist_ok=True)
    optim_steps = 16
    logging_idx = int(len(dataset_train.dataset) / (batch_size * optim_steps))

    #Optimizers
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=0.001, weight_decay=0.01)
    # critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=0.001)

    #schedulers
    # actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=no_epochs, gamma=0.1)
    # critic_scheduler = CosineAnnealingLR(actor_optimizer, T_max=no_epochs, gamma=0.1)

    actor_scheduler = OneCycleLR(actor_optimizer, max_lr=0.005, epochs=no_epochs, steps_per_epoch=optim_steps)
    # critic_scheduler = OneCycleLR(actor_optimizer, T_max=no_epochs, gamma=0.1)

    #Loss function
    loss_function = torch.nn.MSELoss(reduction='none')
    test_loss = torch.nn.MSELoss(reduction='sum')

    actor_best_train_loss = 1e10
    actor_best_test_loss = 1e10

    for epoch_idx in range(no_epochs):
        actor_running_loss = .0
        # critic_running_loss = .0
        avg_max_grad = 0.
        avg_avg_grad = 0.
        for idx, batch in enumerate(iter(dataset_train)):
            input, action, q = unpack_supervised_batch(batch=batch, device=device)

            actor_loss, actor_grad = train(input=input, label=action, net=actor_net, optimizer=actor_optimizer, loss_fn=loss_function)
            # critic_loss, critic_grad = train(input=(action, *input), label=q, net=critic_net, optimizer=critic_optimizer, loss_fn=loss_function)

            avg_max_grad += max([element.max() for element in actor_grad])
            avg_avg_grad += sum([element.mean() for element in actor_grad]) / len(actor_grad)

            actor_running_loss += (actor_loss.item())
            # critic_running_loss += critic_loss.item()

            if idx % logging_idx == logging_idx-1:
                print(f'Actor Epoch: {epoch_idx + 1}, Batch: {idx+1}, Loss: {actor_running_loss/50}, Lr: {actor_scheduler.get_last_lr()}')
                # print(f'Actor Avg Grad: {avg_avg_grad / (idx+1)}, Max Avg Grad: {avg_max_grad / (idx+1)}')
                # print(f'Critic Epoch: {epoch_idx + 1}, Batch: {idx+1}, Loss: {critic_running_loss/50}')
                if (actor_running_loss/50) < actor_best_train_loss:
                    actor_best_train_loss = actor_running_loss/50
                    torch.save(actor_net.state_dict(), f'{path}_train.pt')
                actor_running_loss = 0.0
                # critic_running_loss = 0.0
                avg_max_grad = 0.
                avg_avg_grad = 0.

                actor_scheduler.step()
        print(f'actor_best_train_loss - {actor_best_train_loss}')

        actor_test_loss = .0
        # critic_test_loss = .0
        with torch.no_grad():
            for idx, batch in enumerate(iter(dataset_test)):
                input, action, q = unpack_supervised_batch(batch=batch, device=device)
                actor_pred = actor_net(input)
                # critic_pred = critic_net((action, *input))
                actor_loss = test_loss(actor_pred, action)
                # critic_loss = test_loss(critic_pred.view(-1), q)
                actor_test_loss += actor_loss.item()
                # critic_test_loss += critic_loss.item()

        if (actor_test_loss / len(dataset_test)) < actor_best_test_loss:
            actor_best_test_loss = (actor_test_loss / len(dataset_test))
            torch.save(actor_net.state_dict(), f'{path}_test.pt')

        print(f'Actor test loss {(actor_test_loss/len(dataset_test)):.3f}')
        print(f'Actor best test loss {actor_best_test_loss:.3f}')
        # print(f'Critic test loss {(critic_test_loss/200):.3f}')


def train(input:list, label:torch.Tensor, net:nn.Module, optimizer:torch.optim.Optimizer, loss_fn:torch.nn.MSELoss):
    '''

    :param input:
    :param idx:
    :param net:
    :param optimizer:
    :param device:
    :return:
    '''
    optimizer.zero_grad()
    y_pred = net(input).view(-1)
    loss = loss_fn(y_pred, label.view(-1))
    loss = (loss * (loss / loss.sum())).sum().mean()
    loss.backward()
    grad = [p.detach().cpu().abs() for p in net.parameters()]
    nn.utils.clip_grad_norm_(net.parameters(), 1.)
    optimizer.step()

    return loss, grad


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-s', '--num_steps',
        default=10000,
        type=int,
        dest='num_steps',
        help='Max number of steps per episode, if set to "None" episode will run as long as termiination conditions aren\'t satisfied')

    args = argparser.parse_known_args()
    if len(args) > 1:
        args = args[0]
    try:
        main(args)
    except KeyboardInterrupt:
        print('Interrupted by user! Bye.')





# params = {'batch_size': 32,
#           'shuffle': False,
#           'num_workers': 6}
# p = paths_to_tuples(get_paths()[0])
# # dataset = BufferedDataset(p, transform=transforms.Compose([DepthPreprocess(), ToSupervised()]), buffer_size = 1000)
# # dataset.populate(1000)
# dataset = SimpleDataset(p, transform=transforms.Compose([DepthPreprocess(), ToSupervised()]))
# training_generator = torch.utils.data.DataLoader(dataset, **params)
# iter_training_generator = iter(training_generator)
# a = next(iter_training_generator)