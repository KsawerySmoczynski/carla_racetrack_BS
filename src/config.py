import datetime
import json

import torch

config_dict = json.load(open('../config.json'))

#General
DATE_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M")

#Storage config
STORE_DATA = True
DATA_PATH = '../data'
TENSORBOARD_DATA = f'{DATA_PATH}/tensorboard'
EXPERIMENTS_PATH = f'{DATA_PATH}/experiments'

#World and simulator config
CARLA_IP = config_dict['carla_ip']
FRAMERATE = 30
MAP = 'circut_spa'
INVERT = False
DATA_POINTS = 4

VEHICLES = ['vehicle.mini.cooperst']

#Controller config
IMAGE_DOWNSIZE_FACTOR = 10
IMAGE_SIZE = (80,60)
STEER_BOUNDS = (-1, 1)
THROTTLE_BOUNDS = (-1,1)
#Order of sensors in dict is important for the logging purposes
SENSORS = {
    'depth':True,
    'rgb': True,
    'segmentation': True,
    'collisions': True,
}

# RL config
DEVICE = torch.device('cuda:0')
NEGATIVE_REWARD = 75
NUMERIC_FEATURES = ['state_steer', 'state_gas_brake','distance_2finish','velocity','collisions']
FEATURES_FOR_BATCH = ['step',*NUMERIC_FEATURES,'steer','gas_brake','reward','q']
NO_AGENTS = 1
GAMMA = .9975
LEARNING_RATE = 0.001
NUMBER_OF_EPOCHS = 100
BATCH_SIZE = 20
RANDOM_SEED = 42
