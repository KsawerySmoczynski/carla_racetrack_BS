import datetime
import sys
import time
import argparse
from datetime import date

import numpy as np
import pandas as pd
import carla

#Local imports

from environment import Agent, Environment
from spawn import df_to_spawn_points, numpy_to_transform, to_vehicle_control, set_spectator_above_actor, \
    velocity_to_kmh, location_to_numpy, sensors_config
from control.mpc_control import MPCController
from control.abstract_control import Controller
from tensorboardX import SummaryWriter

#Configs
#TODO Add dynamically generated foldername based on config settings and date.
from config import DATA_PATH, STORE_DATA, FRAMERATE, TENSORBOARD_DATA, ALPHA, \
    DATE_TIME, configure_simulation, SENSORS, VEHICLE

from utils import save_episode_info, calc_distance, tensorboard_log


def run_client(args):

    # Initialize tensorboard -> initialize writer inside run episode so that every
    if args.controller == 'MPC':
        TARGET_SPEED = 90
        STEPS_AHEAD = 10
        writer = SummaryWriter(f'{TENSORBOARD_DATA}/{args.controller}/{args.map}_TS{TARGET_SPEED}_H{STEPS_AHEAD}_FRAMES{args.frames}_{DATE_TIME}',
                               flush_secs=5, max_queue=5)
    else:
        writer = SummaryWriter(f'{TENSORBOARD_DATA}/{args.controller}/{args.map}_FRAMES{args.frames}', flush_secs=5)

    # Connecting to client -> later package it in function which checks if the world is already loaded and if the settings are the same.
    # In order to host more scripts concurrently
    client = configure_simulation(args)

    # create config dict for raport
    #
    # Here let's create data structure which will let us save summary results from each run_episode iteration
    # for ex. status, distance travelled, reward obtained -> may be dataframe, we'll append each row after iteration

    # load spawnpoints from csv -> generate spawn points from notebooks/20200414_setting_points.ipynb
    spawn_points_df = pd.read_csv(f'{DATA_PATH}/spawn_points/{args.map}.csv')
    spawn_points = df_to_spawn_points(spawn_points_df, n=10000, inverse=False) #We keep it here in order to have one way simulation within one script

    # Controller initialization
    if args.controller is 'MPC':
        controller = MPCController(target_speed=TARGET_SPEED, steps_ahead=STEPS_AHEAD, dt=0.05)

    status, actor_dict, env_dict, sensor_data = run_episode(client=client,
                                                            controller=controller,
                                                            spawn_points=spawn_points,
                                                            writer=writer,
                                                            args=args)

    save_episode_info(status, actor_dict, env_dict, sensor_data)


def run_episode(client:carla.Client, controller:Controller, spawn_points:np.array, writer:SummaryWriter, args) -> (str, dict, dict, list):
    '''

    :param actor: vehicle
    :param controller: inherits abstract Controller class
    :param sensors:
    :param way_points:
    :return: status:str ->
             actor_dict -> speed, wheels turn, throttle, reward -> can be taken from actor?
             env_dict -> consecutive locations of actor, distances to closest spawn point, starting spawn point
             array[np.array] -> photos
    '''
    # Create agent object -> delegate everything below to init and configure
    # play_step method returns values from the loop
    NUM_STEPS = args.num_steps
    environment = Environment(client=client)
    world = environment.reset_env(args)
    agent = Agent(world=world, controller=controller, vehicle=args.vehicle,
                  sensors=SENSORS, spawn_points=spawn_points)

    agent.initialize_vehicle()
    spectator = world.get_spectator()
    spectator.set_transform(numpy_to_transform(
        spawn_points[agent.spawn_point_idx-30]))

    # Spawn actor -> how synchronously
    agent_transform = None
    world.tick()
    world.tick()
    # Calculate norm of all cordinates
    while (agent_transform != agent.transform).any():
        agent_transform = agent.transform
        world.tick()

    #INITIALIZE SENSORS
    agent.initialize_sensors()

    # Release handbrake
    world.tick()
    time.sleep(1)# x4? allow controll each 4 frames
    for step in range(NUM_STEPS):  #TODO change to while with conditions
        #Retrieve state and actions
        state = agent.get_state(step)

        # Visdom logging

        #Check if state is terminal
        if state['distance_2finish'] < 30:
            print('lap finished')
            break

        #Apply action
        action = agent.play_step(state) #TODO split to two functions


        #Transit to next state
        world.tick()
        next_state = {
            'velocity': agent.velocity,
            'location': agent.location
        }

        #Receive reward
        reward = environment.calc_reward(points_3D=agent.waypoints, state=state, next_state=next_state,
                                         alpha=ALPHA, step=step)

        #Log
        tensorboard_log(title=DATE_TIME, writer=writer, state=state,
                        action=action, reward=reward, step=step)

        if ((agent.velocity < 20) & (step % 10 == 0)) or (step % 50 == 0):
            set_spectator_above_actor(spectator, agent.transform)

    agent.destroy(data=True)
    del environment

        # Visdom render from depth_data
        # Explore MPC configurations
        # unpack_batch(batch, net, last_val_gamma):
        # calculate for ex. distance and add to separate informative logging structure
        # Uruchomienie 4 instancji środowiska?

    if STORE_DATA:
      pass
    else:
        sensors_data = None



    status, actor_dict, env_dict, sensor_data = str, dict, dict, list

    return status, actor_dict, env_dict, sensor_data


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
        type=int,
        help='Port on the host server (default: 2000)')
    argparser.add_argument(
        '--synchronous',
        metavar='S',
        default=True,
        help='If to run in synchronous mode (currently only this option is avialable)')
    argparser.add_argument(
        '--frames',
        metavar='F',
        default=FRAMERATE,
        type=float,
        help='Number of frames per second, dont set below 10, use with --synchronous flag only')

    #World configs
    argparser.add_argument(
        '--map',
        metavar='M',
        default='circut_spa',
        help='Avialable maps: "circut_spa", "RaceTrack", "Racetrack2". Default: "circut_spa"')
    argparser.add_argument(
        '--vehicle',
        metavar='V',
        default=VEHICLE,
        help='Carla Vehicle blueprint Default: "vehicle.audi.tt"')

    # Simulation
    argparser.add_argument(
        '-s', '--num_steps',
        default=10000,
        type=int,
        dest='num_steps',
        help='Max number of steps per episode, if set to "None" episode will run as long as termiination conditions aren\'t satisfied')


    #Controller configs
    argparser.add_argument(
        '--controller',
        metavar='C',
        default='MPC',
        help='Avialable controllers: "MPC", "NN", Default: "circut_spa"')

    args = argparser.parse_known_args()
    if len(args) > 1:
        args = args[0]

    run_client(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')