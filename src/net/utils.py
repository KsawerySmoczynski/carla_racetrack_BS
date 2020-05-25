import ast
import os
import random
import linecache
import torch
import numpy as np
import pandas as pd
from PIL import Image

from config import SENSORS, FEATURES_FOR_BATCH

to_list = lambda x: ast.literal_eval(x)

def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

def get_paths(path:str='../data/experiments', sensors:dict=SENSORS) -> (dict, list):
    '''
    Method to
    :param path:
    :param sensors:
    :return:
    '''
    sensors_config = '_'.join([sensor*value for sensor, value in sensors.items()])
    paths = [f'{root}/{dir}' for root, dirs, files in os.walk(path) for dir in dirs if sensors_config in dir]
    max_draw = {path:max([int(frame.split('_')[-1][:-4]) for frame in os.listdir(f'{path}/sensors')]) for path in paths}
    dataframes = {path: pd.read_csv(f'{path}/episode_info.csv') for path in max_draw.keys() for path in paths}

    return max_draw, dataframes

def select_batch(paths:dict, states_per_batch:int=32) -> list:
    '''

    :param paths:
    :param states_per_batch:
    :return:
    '''
    batch_indexes = []
    for i in range(states_per_batch):
        experiment = random.choice(list(paths.keys()))
        no_state = random.randint(0, paths[experiment])
        batch_indexes.append((experiment, no_state))

    return batch_indexes

def load_frames(path:str, sensor:str, indexes:list) -> list:
    '''

    :param path:
    :param sensor:
    :param indexes:
    :return:
    '''
    frames = []
    for idx in indexes:
        img = np.array(Image.open(f'{path}/sensors/{sensor}_{idx}.png')).astype(np.uint8)
        frames.append(img)

    return frames

def load_batch(states_indexes:list, dataframes:dict):
    '''
    Element of batch is a state which goes to the preprocessing net method
    :param states_indexes:
    :param dataframes:
    :return:
    '''
    batch = []
    for path, v in states_indexes:
        state = dataframes[path].loc[v, FEATURES_FOR_BATCH].to_dict()
        state = {**state,
                 'depth':load_frames(path, 'depth', to_list(dataframes[path].loc[v,'depth_indexes'])),
                 'rgb':load_frames(path, 'rgb', to_list(dataframes[path].loc[v,'rgb_indexes']))}

        batch.append(state)

    return batch
