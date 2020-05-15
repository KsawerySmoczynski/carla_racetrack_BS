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
VEHICLE = 'vehicle.audi.tt'

#Controller config
IMAGE_DOWNSIZE_FACTOR = 8
STEER_BOUNDS = (-1, 1)
THROTTLE_BOUNDS = (-1,1)
SENSORS = {
    'depth': True,
    'collisions': True,
    'rgb': False
}

# RL config
ALPHA = .9975

def configure_simulation(args):
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)  # seconds

    return client
