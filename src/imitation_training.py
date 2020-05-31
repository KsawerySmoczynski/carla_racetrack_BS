#Script for pretraining reference net
import argparse
import os

import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from config import NUMERIC_FEATURES, DEVICE, DATE_TIME
from net.ddpg_net import DDPGActor, DDPGCritic

np.random.seed(48)
import pandas as pd
import torch
torch.manual_seed(48)
from torchvision import transforms
from torch import multiprocessing as mp, nn
from tensorboardX import SummaryWriter

from net.utils import get_paths, DepthPreprocess, ToSupervised, SimpleDataset, unpack_supervised_batch, get_n_params


#TODO check how pytorch dataloader works and inherit it to build adhoc loading from disk
# https://towardsdatascience.com/deep-learning-model-training-loop-e41055a24b73
# https://towardsdatascience.com/the-false-promise-of-off-policy-reinforcement-learning-algorithms-c56db1b4c79a


def main(args):

    tag = 'depth'
    device = torch.device('cuda:0')

    no_epochs = 3
    batch_size = 128
    depth_shape = [1, 60, 320]

    linear_hidden = 256
    conv_hidden = 64

    #Get train test paths -> later on implement cross val
    steps = get_paths(as_tuples=True, shuffle=True, tag=tag)
    steps_train, steps_test = steps[:int(len(steps)*.8)], steps[int(len(steps)*.2):]

    dataset_train = SimpleDataset(ids=steps_train, batch_size=batch_size, transform=transforms.Compose([DepthPreprocess(), ToSupervised()]))
    dataset_test = SimpleDataset(ids=steps_test, batch_size=batch_size, transform=transforms.Compose([DepthPreprocess(), ToSupervised()]))

    dataloader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': int(mp.cpu_count())} #we've already shuffled paths

    dataset_train = DataLoader(dataset_train, **dataloader_params)
    dataset_test = DataLoader(dataset_test, **dataloader_params)

    #Nets
    # net = DDPGActor(depth_shape=depth_shape, numeric_shape=[len(NUMERIC_FEATURES)], output_shape=[2],
    #                       linear_hidden=linear_hidden, conv_hidden=conv_hidden)
    net = DDPGCritic(actor_out_shape=[2, ], depth_shape=depth_shape, numeric_shape=[len(NUMERIC_FEATURES)],
                            linear_hidden=linear_hidden, conv_hidden=conv_hidden)

    print(len(steps))
    print(net)
    print(get_n_params(net))
    # save path
    net_path = f'../data/models/{DATE_TIME}/{net.name()}'
    os.makedirs(net_path, exist_ok=True)
    optim_steps = 16
    logging_idx = int(len(dataset_train.dataset) / (batch_size * optim_steps))

    writer_train = SummaryWriter(f'{net_path}/train', max_queue=30, flush_secs=5)
    writer_test = SummaryWriter(f'{net_path}/test', max_queue=1, flush_secs=20)

    #Optimizers
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=0.4)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=optim_steps, T_mult=2)

    #Loss function
    loss_function = torch.nn.MSELoss(reduction='none')
    test_loss_function = torch.nn.MSELoss(reduction='sum')

    best_train_loss = 1e10
    best_test_loss = 1e10

    for epoch_idx in range(no_epochs):
        train_loss = .0
        running_loss = .0
        # critic_running_loss = .0
        avg_max_grad = 0.
        avg_avg_grad = 0.
        for idx, batch in enumerate(iter(dataset_train)):
            global_step = ((len(dataset_train.dataset) / batch_size * epoch_idx) + idx)
            input, action, q = unpack_supervised_batch(batch=batch, device=device)

            # loss, loss_mean, grad = train(input=input, label=action, net=net, optimizer=optimizer, loss_fn=loss_function)
            loss, loss_mean, grad = train(input=(action, *input), label=q, net=net, optimizer=optimizer, loss_fn=loss_function)

            avg_max_grad += max([element.max() for element in grad])
            avg_avg_grad += sum([element.mean() for element in grad]) / len(grad)

            running_loss += loss_mean.item()
            train_loss += loss_mean.item()

            writer_train.add_scalar(tag=f'{net.name()}/running_loss',
                                          scalar_value=loss_mean.item(),
                                          global_step=global_step)
            writer_train.add_scalar(tag=f'{net.name()}/max_grad', scalar_value=avg_max_grad,
                                          global_step=global_step)
            writer_train.add_scalar(tag=f'{net.name()}/mean_grad', scalar_value=avg_avg_grad,
                                          global_step=global_step)

            if idx % logging_idx == logging_idx-1:
                print(f'Actor Epoch: {epoch_idx + 1}, Batch: {idx+1}, Loss: {running_loss/logging_idx}, Lr: {scheduler.get_last_lr()[0]}')
                if (running_loss/logging_idx) < best_train_loss:
                    best_train_loss = running_loss/logging_idx
                    torch.save(net.state_dict(), f'{net_path}/train.pt')

                writer_train.add_scalar(tag=f'{net.name()}/lr', scalar_value=scheduler.get_last_lr()[0],
                                              global_step=global_step)
                running_loss = 0.0
                # critic_running_loss = 0.0
                avg_max_grad = 0.
                avg_avg_grad = 0.
                scheduler.step()

        print(f'Actor best train loss for epoch {epoch_idx+1} - {best_train_loss}')
        writer_train.add_scalar(tag=f'{net.name()}/global_loss', scalar_value=train_loss/len(dataset_train),
                                      global_step=(epoch_idx+1))
        test_loss = .0
        # critic_test_loss = .0
        with torch.no_grad():
            for idx, batch in enumerate(iter(dataset_test)):
                input, action, q = unpack_supervised_batch(batch=batch, device=device)
                # pred = net(input)
                pred = net((action, *input))
                # loss = test_loss_function(pred, action)
                loss = test_loss_function(pred.view(-1), q)

                test_loss += loss.item()

        if (test_loss / len(dataset_test)) < best_test_loss:
            best_test_loss = (test_loss / len(dataset_test))
            torch.save(net.state_dict(), f'{net_path}_test.pt')

        print(f'Actor test loss {(test_loss/len(dataset_test)):.3f}')
        print(f'Actor best test loss {best_test_loss:.3f}')
        # print(f'Critic test loss {(critic_test_loss/200):.3f}')
        writer_test.add_scalar(tag=f'{net.name()}/global_loss', scalar_value=(test_loss/len(dataset_test)),
                                     global_step=(epoch_idx + 1))

    writer_train.flush()
    writer_test.flush()


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
    loss_weighted = (loss * (loss / loss.sum())).sum().mean()
    loss_weighted.backward()
    nn.utils.clip_grad_value_(net.parameters(), 1.5)
    grad = [p.detach().cpu().abs() for p in net.parameters()]
    optimizer.step()


    return loss_weighted, loss.mean(), grad


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