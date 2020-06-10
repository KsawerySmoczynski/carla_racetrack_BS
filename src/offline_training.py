#Script for pretraining reference net
import argparse
import json
import os
from subprocess import check_call

import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from config import NUMERIC_FEATURES, DEVICE, DATE_TIME, SENSORS
from control.nn_control import NNController
from net.ddpg_net import DDPGActor, DDPGCritic

np.random.seed(48)
import pandas as pd
import torch
torch.manual_seed(48)
from torchviz import make_dot
from torchvision import transforms
from torch import multiprocessing as mp, nn
from tensorboardX import SummaryWriter

from net.utils import get_paths, DepthPreprocess, ToSupervised, SimpleDataset, unpack_supervised_batch, get_n_params, \
    DepthSegmentationPreprocess

#TODO check how pytorch dataloader works and inherit it to build adhoc loading from disk
# https://towardsdatascience.com/deep-learning-model-training-loop-e41055a24b73
# https://towardsdatascience.com/the-false-promise-of-off-policy-reinforcement-learning-algorithms-c56db1b4c79a

def main(args):

    args = parse_args()
    tag = args.tag
    device = torch.device('cuda:0')

    no_epochs = args.epochs
    batch_size = 128

    linear_hidden = args.linear
    conv_hidden = args.conv

    #Get train test paths -> later on implement cross val
    steps = get_paths(as_tuples=True, shuffle=True, tag=tag)
    steps_train, steps_test = steps[:int(len(steps)*.8)], steps[int(len(steps)*.2):]

    transform = transforms.Compose([DepthSegmentationPreprocess(no_data_points=1), ToSupervised()])

    dataset_train = SimpleDataset(ids=steps_train, batch_size=batch_size, transform=transform, **SENSORS)
    dataset_test = SimpleDataset(ids=steps_test, batch_size=batch_size, transform=transform, **SENSORS)

    dataloader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': int(mp.cpu_count())} #we've already shuffled paths

    dataset_train = DataLoader(dataset_train, **dataloader_params)
    dataset_test = DataLoader(dataset_test, **dataloader_params)

    batch = next(iter(dataset_test))
    action_shape = batch['action'][0].shape
    img_shape = batch['img'][0].shape
    #Nets
    actor_net = DDPGActor(img_shape=img_shape, numeric_shape=[len(NUMERIC_FEATURES)], output_shape=[2],
                          linear_hidden=linear_hidden, conv_hidden=conv_hidden)
    critic_net = DDPGCritic(actor_out_shape=action_shape, img_shape=img_shape, numeric_shape=[len(NUMERIC_FEATURES)],
                            linear_hidden=linear_hidden, conv_hidden=conv_hidden)

    print(len(steps))
    print(actor_net)
    print(get_n_params(actor_net))
    print(critic_net)
    print(get_n_params(critic_net))
    # save path
    actor_net_path = f'../data/models/{DATE_TIME}/{actor_net.name}'
    critic_net_path = f'../data/models/{DATE_TIME}/{critic_net.name}'
    os.makedirs(actor_net_path, exist_ok=True)
    os.makedirs(critic_net_path, exist_ok=True)
    optim_steps = args.optim_steps
    logging_idx = int(len(dataset_train.dataset) / (batch_size * optim_steps))

    actor_writer_train = SummaryWriter(f'{actor_net_path}/train', max_queue=30, flush_secs=5)
    actor_writer_train = SummaryWriter(f'{actor_net_path}/train', max_queue=30, flush_secs=5)
    critic_writer_test = SummaryWriter(f'{critic_net_path}/test', max_queue=1, flush_secs=5)
    critic_writer_test = SummaryWriter(f'{critic_net_path}/test', max_queue=1, flush_secs=5)

    #Optimizers
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=0.001, weight_decay=0.0005)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=0.001, weight_decay=0.0005)

    if args.scheduler == 'cos':
        actor_scheduler = CosineAnnealingWarmRestarts(actor_optimizer, T_0=optim_steps, T_mult=2)
        critic_scheduler = CosineAnnealingWarmRestarts(critic_optimizer, T_0=optim_steps, T_mult=2)
    elif args.scheduler == 'one_cycle':
        actor_scheduler = OneCycleLR(actor_optimizer, max_lr=0.001, epochs=no_epochs,
                                            steps_per_epoch=optim_steps)
        critic_scheduler = OneCycleLR(critic_optimizer, max_lr=0.001, epochs=no_epochs,
                                                    steps_per_epoch=optim_steps)

    controller = NNController(actor_net=actor_net, critic_net=critic_net, no_data_points=1,
                              features=NUMERIC_FEATURES, train=True, device=device)

    #Loss function
    loss_function = torch.nn.MSELoss(reduction='sum')
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
            global_step = int((len(dataset_train.dataset) / batch_size * epoch_idx) + idx)
            batch = unpack_supervised_batch(batch=batch, device=device)
            loss, grad = train(input=batch, label=batch['action'], net=net, optimizer=optimizer, loss_fn=loss_function)
            # loss, grad = train(input=batch, label=batch['q'], net=net, optimizer=optimizer, loss_fn=loss_function)

            avg_max_grad += max([element.max() for element in grad])
            avg_avg_grad += sum([element.mean() for element in grad]) / len(grad)

            running_loss += loss
            train_loss += loss

            writer_train.add_scalar(tag=f'{net.name}/running_loss',
                                          scalar_value=loss,
                                          global_step=global_step)
            writer_train.add_scalar(tag=f'{net.name}/max_grad', scalar_value=avg_max_grad,
                                          global_step=global_step)
            writer_train.add_scalar(tag=f'{net.name}/mean_grad', scalar_value=avg_avg_grad,
                                          global_step=global_step)

            if idx % logging_idx == logging_idx-1:
                print(f'Actor Epoch: {epoch_idx + 1}, Batch: {idx+1}, Loss: {running_loss/logging_idx}, Lr: {scheduler.get_last_lr()[0]}')
                if (running_loss/logging_idx) < best_train_loss:
                    best_train_loss = running_loss/logging_idx
                    torch.save(net.state_dict(), f'{net_path}/train/train.pt')

                writer_train.add_scalar(tag=f'{net.name}/lr', scalar_value=scheduler.get_last_lr()[0],
                                              global_step=global_step)
                running_loss = 0.0
                avg_max_grad = 0.
                avg_avg_grad = 0.
                scheduler.step()

        print(f'{net.name} best train loss for epoch {epoch_idx+1} - {best_train_loss}')
        writer_train.add_scalar(tag=f'{net.name}/global_loss', scalar_value=train_loss/len(dataset_train),
                                      global_step=(epoch_idx+1))
        test_loss = .0
        with torch.no_grad():
            for idx, batch in enumerate(iter(dataset_test)):
                batch = unpack_supervised_batch(batch=batch, device=device)
                pred = net(**batch)
                loss = test_loss_function(pred, batch['action'])
                # loss = test_loss_function(pred.view(-1), batch['q'])

                test_loss += loss

        if (test_loss / len(dataset_test)) < best_test_loss:
            best_test_loss = (test_loss / len(dataset_test))

        torch.save(net.state_dict(), f'{net_path}/test/test_{epoch_idx+1}.pt')

        print(f'{net.name} test loss {(test_loss/len(dataset_test)):.3f}')
        print(f'{net.name} best test loss {best_test_loss:.3f}')
        writer_test.add_scalar(tag=f'{net.name}/global_loss', scalar_value=(test_loss/len(dataset_test)),
                                     global_step=(epoch_idx + 1))

    torch.save(optimizer.state_dict(), f=f'{net_path}/{optimizer.__class__.__name__}.pt')
    torch.save(scheduler.state_dict(), f=f'{net_path}/{scheduler.__class__.__name__}.pt')
    json.dump(vars(args), fp=open(f'{net_path}/args.json', 'w'), sort_keys=True, indent=4)

    batch = next(iter(dataset_test))
    batch = unpack_supervised_batch(batch=batch, device=device)
    y = net(**batch)
    g = make_dot(y, params=dict(net.named_parameters()))
    g.save(filename=f'{DATE_TIME}_{net.name}.dot', directory=net_path)
    check_call(['dot', '-Tpng', '-Gdpi=200', f'{net_path}/{DATE_TIME}_{net.name}.dot', '-o', f'{net_path}/{DATE_TIME}_{net.name}.png'])

    writer_train.flush()
    writer_test.flush()
    writer_train.close()
    writer_test.close()


def train(input:dict, label:torch.Tensor, net:nn.Module, optimizer:torch.optim.Optimizer, loss_fn:torch.nn.MSELoss):
    '''

    :param input:
    :param idx:
    :param net:
    :param optimizer:
    :param device:
    :return:
    '''
    optimizer.zero_grad()
    y_pred = net(**input).view(-1)
    loss = loss_fn(y_pred, label.view(-1))
    loss.backward()
    # loss_weighted = (loss * (loss / (loss.sum()/2))).sum().mean()
    # loss_weighted.backward()
    nn.utils.clip_grad_value_(net.parameters(), 1.5)
    grad = [p.detach().cpu().abs() for p in net.parameters()]
    optimizer.step()

    return loss.detach().cpu(), grad


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-e', '--epochs',
        default=2,
        type=int,
        dest='epochs',
        help='Max number of epochs')
    argparser.add_argument(
        '-c', '--conv',
        default=64,
        type=int,
        dest='conv',
        help='Conv hidden size')
    argparser.add_argument(
        '-l', '--linear',
        default=128,
        type=int,
        dest='linear',
        help='Linear hidden size')
    argparser.add_argument(
        '--optim_steps',
        default=16,
        type=int,
        dest='optim_steps',
        help='Number of optimization steps')
    argparser.add_argument(
        '--scheduler',
        default='cos',
        dest='scheduler',
        help='Number of optimization steps')
    argparser.add_argument(
        '--tag',
        default=None,
        dest='tag',
        help='Filter for dataset')
    args = argparser.parse_known_args()
    if len(args) > 1:
        args = args[0]

    return args


if __name__ == '__main__':
    args = parse_args()

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