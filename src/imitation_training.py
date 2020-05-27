#Script for pretraining reference net
import argparse
import numpy as np
from torch.utils.data import DataLoader

from config import NUMERIC_FEATURES
from net.ddpg_net import DDPGActor, DDPGCritic

np.random.seed(0)
import pandas as pd
import torch
torch.manual_seed(0)
from torchvision import transforms
from torch import multiprocessing as mp

from net.utils import get_paths, DepthPreprocess, ToSupervised, SimpleDataset, unpack_supervised_batch


#TODO check how pytorch dataloader works and inherit it to build adhoc loading from disk
# https://towardsdatascience.com/deep-learning-model-training-loop-e41055a24b73
# https://towardsdatascience.com/the-false-promise-of-off-policy-reinforcement-learning-algorithms-c56db1b4c79a

#1. Load data indices
#2. Choose indices for train and test
#3. Shuffle indices and prepare for batching
#4. Initialize nets and optimizers
#5. Start looping
#       5.1 Load in memory one epoch batches batchsize = 64, epoch=128batches
#       equals to 8192 steps of simulation it means roughly 32768 frames.
#       We'll take one in memory epoch.
#       5.2 Feed batches calculate losses and make step after n=?? batches
#       5.3 Write data to tensorboard and save net if its best epoch



def main(args):

    device = torch.device('cuda:0')
    no_epochs = 2
    batch_size = 64

    #Get train test paths -> later on implement cross val
    steps = get_paths(as_tuples=True, shuffle=True)
    steps_train, steps_test = steps[:int(len(steps)*.8)][:20000], steps[int(len(steps)*.8):]

    dataset_train = SimpleDataset(steps_train, batch_size=batch_size, transform=transforms.Compose([DepthPreprocess(), ToSupervised()]))
    dataset_test = SimpleDataset(steps_test, batch_size=batch_size, transform=transforms.Compose([DepthPreprocess(), ToSupervised()]))

    dataloader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': int(mp.cpu_count())} #we've already shuffled paths

    dataset_train = DataLoader(dataset_train, **dataloader_params)
    dataset_test = DataLoader(dataset_test, **dataloader_params)

    #Nets
    actor_net = DDPGActor(depth_shape=[1, 60, 320], numeric_shape=[len(NUMERIC_FEATURES)], output_shape=[2])
    # critic_net = DDPGCritic(actor_out_shape=[2, ], depth_shape=[1, 60, 320], numeric_shape=[len(NUMERIC_FEATURES)])

    #Optimizers
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=0.001)
    # critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=0.001)

    #Loss function
    loss_function = torch.nn.MSELoss(reduction='sum')

    for epoch_idx in range(no_epochs):
        actor_running_loss = .0
        # critic_running_loss = .0
        for idx, batch in enumerate(iter(dataset_train)):
            input, action, q = unpack_supervised_batch(batch=batch, device=device)

            actor_optimizer.zero_grad()
            # critic_optimizer.zero_grad()

            actor_pred = actor_net(input)
            # critic_pred = critic_net((action, *input))

            actor_loss = loss_function(actor_pred, action)
            # critic_loss = loss_function(critic_pred.view(-1), q)

            actor_loss.backward()
            # critic_loss.backward()

            actor_optimizer.step()
            # critic_optimizer.step()

            actor_running_loss += actor_loss.item()
            # critic_running_loss += critic_loss.item()

            if idx % 10 == 0:  # print every 2000 mini-batches
                print(f'Actor Epoch: {epoch_idx + 1}, Batch: {idx+1}, Loss: {actor_running_loss/10}')
                # print(f'Critic Epoch: {epoch_idx + 1}, Batch: {idx+1}, Loss: {critic_running_loss/10}')
                actor_running_loss = 0.0
                # critic_running_loss = 0.0

        test_loss = .0
        with torch.no_grad():
            for idx, batch in enumerate(iter(dataset_test)):
                if idx > 200:
                    break
                input, action, q = unpack_supervised_batch(batch=batch, device=device)
                actor_pred = actor_net(input)
                actor_loss = loss_function(actor_pred, action)
                test_loss += actor_loss.item()

        print(f'Test loss {(test_loss/200):.3f}')

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