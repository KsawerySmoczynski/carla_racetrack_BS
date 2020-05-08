import sys
import os
sys.path.append(f'{os.getcwd()}/src')
import json

config_dict = json.load(open('config.json'))

STEER_BOUNDS = (-1, 1)
THROTTLE_BOUNDS = (-1,1)
CARLA_IP = config_dict['carla_ip']