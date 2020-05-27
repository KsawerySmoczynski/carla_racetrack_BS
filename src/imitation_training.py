#Script for pretraining reference net
import argparse
import numpy as np
from torch.utils.data import DataLoader

from net.ddpg_net import DDPGActor, DDPGCritic

np.random.seed(0)
import pandas as pd
import torch
torch.manual_seed(0)
from torchvision import transforms
from torch import multiprocessing as mp

from net.utils import get_paths, paths_to_tuples, DepthPreprocess, ToSupervised, SimpleDataset, Batcher


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

    no_epochs = 10
    batch_size = 1
    epoch_size = 128

    #Get train test paths -> later on implement cross val
    steps = paths_to_tuples(get_paths()[0])
    steps_random = [steps[i] for i in np.random.permutation(range(len(steps)))]
    # steps_random = np.random.permutation(steps)
    steps_train, steps_test = steps_random[:int(len(steps_random)*.8)], steps_random[int(len(steps_random)*.8):]

    dataset_train = SimpleDataset(steps_train, batch_size=32, transform=transforms.Compose([DepthPreprocess(), ToSupervised()]))
    dataset_test = SimpleDataset(steps_test, batch_size=32, transform=transforms.Compose([DepthPreprocess(), ToSupervised()]))

    dataloader_params =  {'batch_size': batch_size * epoch_size, 'shuffle': False, 'num_workers': int(mp.cpu_count()/2)} #we've already shuffled paths

    dataset_train = iter(DataLoader(dataset_train, **dataloader_params))
    dataset_test = iter(DataLoader(dataset_test, **dataloader_params))

    #Nets
    actor_net = DDPGActor(depth_shape=[1], numeric_shape=[4], output_shape=[2])
    critic_net = DDPGCritic(depth_shape=[1], numeric_shape=[4], output_shape=[1])

    #Optimizers

    for idx in range(no_epochs):
        for idx, buffer in enumerate(dataset_train):
            buffer = Batcher(minibatch=buffer, batch_size=batch_size, supervised=True)
            for batch in buffer:
                inputs, actions, q = batch








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





params = {'batch_size': 128,
          'shuffle': False,
          'num_workers': 6}
p = paths_to_tuples(get_paths()[0])
# dataset = BufferedDataset(p, transform=transforms.Compose([DepthPreprocess(), ToSupervised()]), buffer_size = 1000)
# dataset.populate(1000)
dataset = SimpleDataset(p, transform=transforms.Compose([DepthPreprocess(), ToSupervised()]))
training_generator = torch.utils.data.DataLoader(dataset, **params)
iter_training_generator = iter(training_generator)
a = next(iter_training_generator)