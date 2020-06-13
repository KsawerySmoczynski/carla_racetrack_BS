#Script for pretraining reference net
import argparse
import json
import os
from subprocess import check_call

import numpy as np
from torch.utils.data import DataLoader

from config import NUMERIC_FEATURES, DEVICE, DATE_TIME, SENSORS
from net.ddpg_net import DDPGActor, DDPGCritic

np.random.seed(48)
import pandas as pd
import torch
torch.manual_seed(48)
from torchviz import make_dot
from torchvision import transforms
from torch import multiprocessing as mp, nn
from tensorboardX import SummaryWriter

from net.utils import get_paths, DepthPreprocess, ToSupervised, SimpleDataset, unpack_batch, get_n_params, \
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
    actor_net_path = f'../data/models/offline/{DATE_TIME}/{actor_net.name}'
    critic_net_path = f'../data/models/offline/{DATE_TIME}/{critic_net.name}'
    os.makedirs(actor_net_path, exist_ok=True)
    os.makedirs(critic_net_path, exist_ok=True)
    optim_steps = args.optim_steps
    logging_idx = int(len(dataset_train.dataset) / (batch_size * optim_steps))

    actor_writer_train = SummaryWriter(f'{actor_net_path}/train', max_queue=30, flush_secs=5)
    critic_writer_train = SummaryWriter(f'{critic_net_path}/train', max_queue=1, flush_secs=5)
    actor_writer_test = SummaryWriter(f'{actor_net_path}/test', max_queue=30, flush_secs=5)
    critic_writer_test = SummaryWriter(f'{critic_net_path}/test', max_queue=1, flush_secs=5)

    #Optimizers
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=0.001, weight_decay=0.0005)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=0.001, weight_decay=0.0005)

    #Loss function
    loss_function = torch.nn.MSELoss(reduction='sum')

    actor_best_train_loss = 1e10
    critic_best_train_loss = 1e10
    actor_best_test_loss = 1e10
    critic_best_test_loss = 1e10

    for epoch_idx in range(no_epochs):
        actor_train_loss = .0
        critic_train_loss = .0
        actor_running_loss = .0
        critic_running_loss = .0
        actor_avg_max_grad = .0
        critic_avg_max_grad = .0
        actor_avg_avg_grad = .0
        critic_avg_avg_grad = .0
        for idx, batch in enumerate(iter(dataset_train)):
            global_step = int((len(dataset_train.dataset) / batch_size * epoch_idx) + idx)
            batch = unpack_batch(batch=batch, device=device)
            actor_loss, critic_loss, actor_grad, critic_grad = train_rl(batch=batch, actor_net=actor_net, critic_net=critic_net,
                                  actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer, loss_fn=loss_function)


            actor_avg_max_grad  += max([element.max() for element in actor_grad])
            critic_avg_max_grad += max([element.max() for element in critic_grad])
            actor_avg_avg_grad  += sum([element.mean() for element in actor_grad]) / len(actor_grad)
            critic_avg_avg_grad += sum([element.mean() for element in critic_grad]) / len(critic_grad)

            actor_running_loss += actor_loss
            critic_train_loss += critic_loss
            actor_train_loss += actor_loss
            critic_running_loss += critic_loss

            actor_writer_train.add_scalar(tag=f'{actor_net.name}/running_loss',
                                          scalar_value=actor_loss,
                                          global_step=global_step)
            actor_writer_train.add_scalar(tag=f'{actor_net.name}/max_grad', scalar_value=actor_avg_max_grad,
                                          global_step=global_step)
            actor_writer_train.add_scalar(tag=f'{actor_net.name}/mean_grad', scalar_value=actor_avg_avg_grad,
                                          global_step=global_step)

            critic_writer_train.add_scalar(tag=f'{critic_net.name}/running_loss',
                                          scalar_value=critic_loss,
                                          global_step=global_step)
            critic_writer_train.add_scalar(tag=f'{critic_net.name}/max_grad', scalar_value=critic_avg_max_grad,
                                          global_step=global_step)
            critic_writer_train.add_scalar(tag=f'{critic_net.name}/mean_grad', scalar_value=critic_avg_avg_grad,
                                          global_step=global_step)

            if idx % logging_idx == logging_idx-1:
                print(f'Actor Epoch: {epoch_idx + 1}, Batch: {idx+1}, Loss: {actor_running_loss/logging_idx}')
                print(f'Critic Epoch: {epoch_idx + 1}, Batch: {idx+1}, Loss: {critic_running_loss/logging_idx}')
                if (critic_running_loss/logging_idx) < critic_best_train_loss:
                    critic_best_train_loss = critic_running_loss/logging_idx
                    torch.save(actor_net.state_dict(), f'{actor_net_path}/train/train.pt')
                    torch.save(critic_net.state_dict(), f'{critic_net_path}/train/train.pt')

                actor_running_loss = .0
                actor_avg_max_grad = .0
                actor_avg_avg_grad = .0
                critic_running_loss = .0
                critic_avg_max_grad = .0
                critic_avg_avg_grad = .0

        print(f'{critic_net.name} best train loss for epoch {epoch_idx+1} - {critic_best_train_loss}')
        actor_writer_train.add_scalar(tag=f'{actor_net.name}/global_loss', scalar_value=actor_train_loss/len(dataset_train),
                                      global_step=(epoch_idx+1))
        critic_writer_train.add_scalar(tag=f'{critic_net.name}/global_loss', scalar_value=critic_train_loss/len(dataset_train),
                                      global_step=(epoch_idx+1))
        actor_test_loss = .0
        critic_test_loss = .0
        with torch.no_grad():
            for idx, batch in enumerate(iter(dataset_test)):
                batch = unpack_batch(batch=batch, device=device)
                q_pred = critic_net(**batch)
                action_pred = actor_net(**batch)
                critic_test_loss = loss_function(q_pred.view(-1), batch['q'])
                actor_test_loss = loss_function(action_pred, batch['action'])


                critic_test_loss += critic_test_loss
                actor_test_loss += actor_test_loss

        if (critic_test_loss / len(dataset_test)) < critic_best_test_loss:
            critic_best_test_loss = (critic_test_loss / len(dataset_test))
        if (actor_test_loss / len(dataset_test)) < actor_best_test_loss:
            actor_best_test_loss = (actor_test_loss / len(dataset_test))

        torch.save(critic_net.state_dict(), f'{critic_net_path}/test/test_{epoch_idx+1}.pt')
        torch.save(actor_net.state_dict(), f'{actor_net_path}/test/test_{epoch_idx+1}.pt')

        print(f'{critic_net.name} test loss {(critic_test_loss/len(dataset_test)):.3f}')
        print(f'{actor_net.name} test loss {(actor_test_loss/len(dataset_test)):.3f}')
        print(f'{critic_net.name} best test loss {critic_best_test_loss:.3f}')
        print(f'{actor_net.name} best test loss {actor_best_test_loss:.3f}')

        critic_writer_test.add_scalar(tag=f'{critic_net.name}/global_loss', scalar_value=(critic_test_loss/len(dataset_test)),
                                     global_step=(epoch_idx + 1))
        actor_writer_test.add_scalar(tag=f'{actor_net.name}/global_loss', scalar_value=(actor_test_loss/len(dataset_test)),
                                     global_step=(epoch_idx + 1))

    torch.save(actor_optimizer.state_dict(), f=f'{actor_net_path}/{actor_optimizer.__class__.__name__}.pt')
    torch.save(critic_optimizer.state_dict(), f=f'{critic_net_path}/{critic_optimizer.__class__.__name__}.pt')
    json.dump(vars(args), fp=open(f'{actor_net_path}/args.json', 'w'), sort_keys=True, indent=4)
    json.dump(vars(args), fp=open(f'{critic_net_path}/args.json', 'w'), sort_keys=True, indent=4)

    batch = next(iter(dataset_test))
    batch = unpack_batch(batch=batch, device=device)

    #Actor architecture save
    y = actor_net(**batch)
    g = make_dot(y, params=dict(actor_net.named_parameters()))
    g.save(filename=f'{DATE_TIME}_{actor_net.name}.dot', directory=actor_net_path)
    check_call(['dot', '-Tpng', '-Gdpi=200', f'{actor_net_path}/{DATE_TIME}_{actor_net.name}.dot', '-o',
                f'{actor_net_path}/{DATE_TIME}_{actor_net.name}.png'])

    #Critic architecture save
    y = critic_net(**batch)
    g = make_dot(y, params=dict(critic_net.named_parameters()))
    g.save(filename=f'{DATE_TIME}_{critic_net.name}.dot', directory=critic_net_path)
    check_call(['dot', '-Tpng', '-Gdpi=200', f'{critic_net_path}/{DATE_TIME}_{critic_net.name}.dot', '-o',
                f'{critic_net_path}/{DATE_TIME}_{critic_net.name}.png'])

    actor_writer_train.flush()
    actor_writer_test.flush()
    actor_writer_train.close()
    actor_writer_test.close()
    critic_writer_train.flush()
    critic_writer_test.flush()
    critic_writer_train.close()
    critic_writer_test.close()


def train_rl(batch, actor_net:nn.Module, critic_net:nn.Module, actor_optimizer:torch.optim.Optimizer,
          critic_optimizer:torch.optim.Optimizer, loss_fn:torch.nn.MSELoss):
    '''

    :param batch:
    :param actor_net:
    :param critic_net:
    :param actor_optimizer:
    :param critic_optimizer:
    :param loss_fn:
    :return:
    '''
    critic_optimizer.zero_grad()
    q_pred = critic_net(**batch).view(-1)
    critic_loss = loss_fn(q_pred, batch['q'].view(-1))
    critic_loss.backward()
    nn.utils.clip_grad_value_(critic_net.parameters(), 1.5)
    critic_grad = [p.detach().cpu().abs() for p in critic_net.parameters()]
    critic_optimizer.step()

    actor_optimizer.zero_grad()
    action = actor_net(**batch)
    batch['action'] = action
    actor_loss = -critic_net(**batch)
    actor_loss = actor_loss.mean()
    actor_loss.backward()
    nn.utils.clip_grad_value_(actor_net.parameters(), 1.5)
    actor_grad = [p.detach().cpu().abs() for p in actor_net.parameters()]
    actor_optimizer.step()

    return actor_loss, critic_loss, actor_grad, critic_grad




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