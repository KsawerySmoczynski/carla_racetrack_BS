import json
import carla

config_dict = json.load(open('../config.json'))

STORE_DATA = True
DATA_PATH = '../data'

STEER_BOUNDS = (-1, 1)
THROTTLE_BOUNDS = (-1,1)
CARLA_IP = config_dict['carla_ip']

IMAGE_DOWNSIZE_FACTOR = 8

FRAMERATE = 20

def toggle_world(world:carla.World, frames:int=FRAMERATE):
    settings = world.get_settings()
    settings.synchronous_mode = not settings.synchronous_mode
    settings.fixed_delta_seconds = abs(float(settings.fixed_delta_seconds or 0) - 1/frames)
    world.apply_settings(settings)
