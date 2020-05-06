import os
import sys
import time
import argparse
import math
sys.path.append(['..'])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import carla

#Local imports
from src.spawn import df_to_spawn_points

#Configs
map_idx = 2
MAP = ['circut_spa', 'RaceTrack', 'RaceTrack2']

spawn_points_df = pd.read_csv(f'data/spawn_points/{MAP[map_idx]}.csv')

#Connecting to client
client = carla.Client('localhost', 2000)
client.set_timeout(5.0) # seconds

world = client.load_world(MAP[map_idx])
blueprint_library = world.get_blueprint_library()
vehicle = blueprint_library.filter('*aud*')[0]
spawn_points = df_to_spawn_points(data=spawn_points_df)
spectator = world.get_spectator()

def run_client(args):
    # create client -> config client does that
    #   check if loaded map is target map, if true proceed if false try to load desired world if other actor on map raise error -> or write error and close
    #
    # load world desired condition -> config client does that
    #
    # create config dict for raport
    #
    # load spawnpoints from csv
    #
    # initialize proper controller
    #
    # episodes loop
    #   randomly choose start point, flip spawn points so that last one will be prior to starting one -> way_points
    #   create actor and attach sensors -> pass it as a batch
    #
    #   run_episode -> status, imgs, states-action dicts, distance
    #
    #       #saving state-action dicts + imgs in order to feed DQN in off-policy mode.
    # save info to report

    pass

def run_episode(controller, sensors, way_points) -> (int, list, dict, float):

    pass


def main():
    #parse args
    #   - GPUS
    #   - CONTROLLER
    #   - LOGGING
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    args = argparser.parse_args()

    #asynchronous config client for world parameters and loading

    # run client from args -> try pool -> multiprocessing for different GPUS. Clients cant affect world settings,
    #  only apply control to their vehicles
    run_client(args)
    pass

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')