import os
import sys
import time
import argparse
import math
sys.path.append(f'{os.getcwd()}/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import carla

#Local imports
from spawn import df_to_spawn_points, to_transform

#Configs
from src.config import CARLA_IP

# spectator = world.get_spectator()

def run_client(args):
    # create client -> config client does that
    #   check if loaded map is target map, if true proceed if false try to load desired world if other actor on map raise error -> or write error and close

    # Connecting to client
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)  # seconds
    # load world desired condition -> config client does that
    world = client.load_world(args.map)
    if args.synchronous:
        settings = world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        world.apply_settings(settings)
    blueprint_library = world.get_blueprint_library()
    vehicle = blueprint_library.filter('*aud*')[0] # -> change it to be parametric, maybe shuffle each time to add robustness

    # create config dict for raport
    #
    # Here let's create data structure which will let us save summary results from each run_episode iteration
    # for ex. status, distance travelled, reward obtained -> may be dataframe, we'll append each row after iteration


    # load spawnpoints from csv -> generate spawn points from notebooks/20200414_setting_points.ipynb
    spawn_points_df = pd.read_csv(f'data/spawn_points/{args.map}.csv')
    spawn_points_np = df_to_spawn_points(spawn_points_df)
    spawn_points = [to_transform(sp) for sp in spawn_points_np]
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

def run_episode(actor, controller, sensors, way_points) -> dict:
    '''

    :param actor:
    :param controller:
    :param sensors:
    :param way_points:
    :return: actor_dict -> speed, wheels turn, throttle, reward -> can be taken from actor?
             env_dict -> consecutive locations of actor, distances to closest spawn point, starting spawn point
             array[np.array] -> photos
    '''

    for step in range(steps_per_episode):


    pass


def main():
    #parse args
    #   - GPUS
    #   - CONTROLLER
    #   - LOGGING
    argparser = argparse.ArgumentParser()
    # Simulator configs
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '--port',
        metavar='P',
        default=2000,
        help='Port on the host server (default: 2000)')
    argparser.add_argument(
        '--synchronous',
        metavar='S',
        default=True,
        help='If to run in synchronous mode (currently only this option is avialable)')

    #World configs

    argparser.add_argument(
        '--map',
        metavar='M',
        default='circut_spa',
        help='Avialable maps: "circut_spa", "RaceTrack", "Racetrack2". Default: "circut_spa"')
    args = argparser.parse_known_args()
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