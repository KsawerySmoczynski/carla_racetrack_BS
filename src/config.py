import datetime
import json
import carla

config_dict = json.load(open('../config.json'))

#General
DATE_TIME = datetime.datetime.now().strftime('%Y%m%d_%H%M')

#Storage config
STORE_DATA = True
DATA_PATH = '../data'
TENSORBOARD_DATA = '../data/tensorboard'

#World and simulator config
CARLA_IP = config_dict['carla_ip']
FRAMERATE = 30
MAP = 'circut_spa'
VEHICLE = 'vehicle.audi.tt'

#Controller config
IMAGE_DOWNSIZE_FACTOR = 8
STEER_BOUNDS = (-1, 1)
THROTTLE_BOUNDS = (-1,1)
SENSORS = {
    'depth': True,
    'collisions': True,
    'rgb': True
}

# RL config
ALPHA = .9975
LEARNING_RATE = 0.001
BATCH_SIZE = 20
EXP_BUFFER = 4

def configure_simulation(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)  # seconds

    return client
