import ast
import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

from config import SENSORS, FEATURES_FOR_BATCH, DEVICE, NUMERIC_FEATURES


to_list = lambda x: ast.literal_eval(x)
img_to_pil = lambda img: Image.fromarray(img, 'RGB').convert('L').filter(ImageFilter.FIND_EDGES)

def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

def weights_init(m, cuda:bool = True):
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


def get_paths(path:str='../data/experiments', sensors:dict=SENSORS, as_tuples:bool=False, shuffle:bool=False) -> dict:
    '''
    Method to
    :param path:
    :param sensors:
    :return:
    '''
    sensors_config = '_'.join([sensor*value for sensor, value in sensors.items()])
    paths = [f'{root}/{dir}' for root, dirs, files in os.walk(path) for dir in dirs if sensors_config in dir]
    paths = [path for path in paths if 'q' in pd.read_csv(f"{path}/episode_info.csv", nrows=1).columns]
    steps = {path:max([int(frame.split('_')[-1][:-4]) for frame in os.listdir(f'{path}/sensors')]) for path in paths}
    # dataframes = {path: pd.read_csv(f'{path}/episode_info.csv') for path in max_draw.keys() for path in paths} if dfs else None
    if as_tuples:
        steps =  [(path, step) for path, steps_q in steps.items() for step in range(steps_q)]
        if shuffle:
            steps = [steps[i] for i in np.random.permutation(range(len(steps)))]

    return steps


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


def load_frames(path:str, sensor:str, indexes:list, convert=lambda x:x) -> list:
    '''

    :param path:
    :param sensor:
    :param indexes:
    :return:
    '''
    frames = []
    for idx in indexes:
        img = convert(Image.open(f'{path}/sensors/{sensor}_{idx}.png'))
        img = np.array(img).astype(np.uint8)
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


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def unpack_supervised_batch(batch, device=DEVICE):
    '''

    :param batch:
    :param device:
    :return:
    '''
    input, action, q = batch
    return (input[0].float().to(device), input[1].float().to(device)), action.float().to(device), q.float().to(device)


class OldToSupervised(object):
    def __call__(self, sample):
        inputs = [np.array([sample['data'][data] for data in ['distance_2finish', 'velocity', 'collisions']]), sample['data']['depth']]  # + [[np.Nan]]  # - how to add None
        inputs = tuple(inputs)
        # inputs = np.array([sample['data'][key] for key in list(sample['data'].keys()) if key not in ['q', 'reward', 'step']])
        actions = np.array([sample['data']['steer'], sample['data']['gas_brake']])
        q = sample['data']['q']

        return {'inputs': inputs, 'actions': actions, 'q': q}


class ToSupervised(object):
    def __init__(self, features:list=NUMERIC_FEATURES):
        self.features = features
    def __call__(self, sample):
        input = [np.array([sample['data'][data] for data in self.features]), sample['data']['depth']]  # + [[np.Nan]]  # - how to add None
        input = tuple(input)
        action = np.array([sample['data']['steer'], sample['data']['gas_brake']])
        q = sample['data']['q']

        return (input, action, q)


class ToReinforcement(object):
    def __call__(self, sample):
        pass


class ImgsPreprocess(object):
    def __call__(self, sample):
        pass


class DepthPreprocess(object):
    def __call__(self, sample):
        convert = lambda x: x.convert('L').filter(ImageFilter.FIND_EDGES)
        sample['data']['depth'] = load_frames(path=sample['item'][0], sensor='depth',
                                              indexes=to_list(sample['data']['depth_indexes']), convert=convert)
        del sample['data']['depth_indexes']
        sample['data']['depth'] = np.concatenate([img.reshape(1, img.shape[0], img.shape[1])
                                                  for img in sample['data']['depth']], axis=2) / 255.

        sample['data']['depth'] = sample['data']['depth'] - sample['data']['depth'].mean()

        return sample


class RgbPreprocess(object):
    def __call__(self, sample):
        pass


class SimpleDataset(Dataset):
    def __init__(self, ids, features:list=FEATURES_FOR_BATCH, depth:bool=True, rgb:bool=False, transform:list=None, batch_size:int=32):
        '''
        Klasa przechowuje informacje o ścieżkach i stepach należących do datasetu
        '''
        assert isinstance(ids, (list, tuple, type(np.array([])))), 'Ids should be an array of tuples (dir,  step)'
        assert isinstance(ids[0], (list, tuple, type(np.array([])))) and len(ids[0]) == 2, 'Element of ids should be an array, list or tuple containing 2 elements'

        # for idx in [random.randint(0, len(ids)) for i in range(len(ids) % batch_size)]:
        #     ids.pop(idx)
        self.ids = ids[:-(len(ids) % batch_size)] if ((len(ids) % batch_size) != 0) else ids
        self.dfs = list(set([id[0] for id in ids]))
        self.dfs = {directory:pd.read_csv(f'{directory}/episode_info.csv') for directory in self.dfs}
        self.features = features + list(filter(lambda x: len(x) > 1, ['depth_indexes'*depth, 'rgb_indexes'*rgb]))
        self.transform = transform if transform else lambda x: x

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item_idx):

        data = self.dfs[self.ids[item_idx][0]].loc[self.ids[item_idx][1], self.features].to_dict()
        sample = {'item': self.ids[item_idx], 'data': data}

        return self.transform(sample)

# https://github.com/Shmuma/ptan/blob/049ff123f5967eaeeaa268684e13e5aec5029d9f/ptan/experience.py#L327
class BufferedDataset(Dataset):
    def __init__(self, ids, features: list = FEATURES_FOR_BATCH, depth: bool = True, rgb: bool = False,
                 transform=None, buffer_size:int=100_000):
        '''
        Klasa przechowuje informacje o ścieżkach i stepach należących do datasetu
        '''
        assert isinstance(ids, (list, np.array)), 'Ids should be an array of tuples (dir,  step)'
        assert isinstance(ids[0], (list, tuple)) and len(
            ids[0]) == 2, 'Element of ids should be an array, list or tuple containing 2 elements'

        self.ids = ids
        self.dfs = list(set([id[0] for id in ids]))
        self.dfs = {directory: pd.read_csv(f'{directory}/episode_info.csv') for directory in self.dfs}
        # [path for path, df in dataset.dfs.items() if 'q' not in df.columns] dont use csvs without q
        self.features = features + list(filter(lambda x: len(x) > 1, ['depth_indexes' * depth, 'rgb_indexes' * rgb]))
        self.transform = transform if transform else lambda x: x
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item_idx):

        return self.buffer[item_idx]

    def populate(self, n:int=1):
        '''
        Load n-samples to buffer
        :return:
        '''
        for i in range(n):
            item_idx = np.random.randint(0, len(self.ids))
            data = self.dfs[self.ids[item_idx][0]].loc[self.ids[item_idx][1], self.features].to_dict()
            sample = {'item': self.ids[item_idx], 'data': data}
            self.ids.pop(item_idx)
            self._add(self.transform(sample))

    def add_to_buffer(self, sample):
        #transform 2 current buffer form with transformations
        self._add(self.transform(sample))

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def get_batch(self, batch_size=32):
        return self.buffer[np.random.randint(0, len(self), size=(batch_size))]


class BufferedAdHocDataset(Dataset):
    def __init__(self, ids, features: list = FEATURES_FOR_BATCH, depth: bool = True, rgb: bool = False,
                 transform=None, buffer_size:int=100_000):
        '''
        #TODO klasa trzyma w bufferze ścieżki załadowane z dysku, dopychamy do niej nowe (dir, step) po zapisie np. 1000 stepów wypychając 1000 najstarszych entry.
        #Dataloader trzyma referencje a nie obiekt więc będzie updatował na bierząco
        '''
        assert isinstance(ids, (list, np.array)), 'Ids should be an array of tuples (dir,  step)'
        assert isinstance(ids[0], (list, tuple)) and len(
            ids[0]) == 2, 'Element of ids should be an array, list or tuple containing 2 elements'

        self.ids = ids
        self.dfs = list(set([id[0] for id in ids]))
        self.dfs = {directory: pd.read_csv(f'{directory}/episode_info.csv') for directory in self.dfs}
        # [path for path, df in dataset.dfs.items() if 'q' not in df.columns] dont use csvs without q
        self.features = features + list(filter(lambda x: len(x) > 1, ['depth_indexes' * depth, 'rgb_indexes' * rgb]))
        self.transform = transform if transform else lambda x: x
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item_idx):

        return self.buffer[item_idx]

    def populate(self, n:int=1):
        '''
        Load n-samples to buffer
        :return:
        '''
        for i in range(n):
            item_idx = np.random.randint(0, len(self.ids))
            data = self.dfs[self.ids[item_idx][0]].loc[self.ids[item_idx][1], self.features].to_dict()
            sample = {'item': self.ids[item_idx], 'data': data}
            self.ids.pop(item_idx)
            self._add(self.transform(sample))

    def add_to_buffer(self, sample):
        #transform 2 current buffer form with transformations
        self._add(self.transform(sample))

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def get_batch(self, batch_size=32):
        return self.buffer[np.random.randint(0, len(self), size=(batch_size))]



