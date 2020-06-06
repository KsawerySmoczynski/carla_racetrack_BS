import argparse
import json
import time
import datetime

import numpy as np
import pandas as pd
import carla

#Local imports
import torch
from control.nn_control import NNController
from environment import Agent, Environment
from net.ddpg_net import DDPGActor, DDPGCritic
from spawn import df_to_spawn_points, numpy_to_transform, set_spectator_above_actor, configure_simulation, \
    to_vehicle_control
from control.mpc_control import MPCController
from control.abstract_control import Controller
from tensorboardX import SummaryWriter

#Configs
#TODO Add dynamically generated foldername based on config settings and date.
from config import DATA_PATH, FRAMERATE, TENSORBOARD_DATA, GAMMA, \
    DATE_TIME, SENSORS, VEHICLES, CARLA_IP, MAP, NEGATIVE_REWARD, NUMERIC_FEATURES, DATA_POINTS

from utils import save_info, update_Qvals


def main():
    argparser = argparse.ArgumentParser()
    # Simulator configs
    argparser.add_argument(
        '--host',
        metavar='H',
        default=CARLA_IP,
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
        default=MAP,
        help='Avialable maps: "circut_spa", "RaceTrack", "Racetrack2". Default: "circut_spa"')
    argparser.add_argument(
        '--invert',
        default=False,
        type=bool,
        help='Inverts the track')

    argparser.add_argument(
        '--vehicle',
        metavar='V',
        default=0,
        type=int,
        dest='vehicle',
        help=f'Carla Vehicle blueprint, choose with integer. Avialable: {VEHICLES}')

    # Simulation
    argparser.add_argument(
        '-e', '--episodes',
        default=1,
        type=int,
        dest='episodes',
        help='Number of episodes')

    argparser.add_argument(
        '-s', '--num_steps',
        default=5000,
        type=int,
        dest='num_steps',
        help='Max number of steps per episode, if set to "None" episode will run as long as termiination conditions aren\'t satisfied')

    #Controller configs
    argparser.add_argument(
        '--controller',
        metavar='C',
        default='MPC',
        help='Avialable controllers: "MPC", "NN", Default: "MPC"')

    argparser.add_argument(
        '--speed',
        default=90,
        type=int,
        dest='speed',
        help='Target speed for mpc')

    argparser.add_argument(
        '--steps_ahead',
        default=10,
        type=int,
        dest='steps_ahead',
        help='steps 2calculate ahead for mpc')

    argparser.add_argument(
        '-c', '--conv',
        default=64,
        type=int,
        dest='conv',
        help='Conv hidden size')

    argparser.add_argument(
        '-l', '--linear',
        default=128,
        type=int,
        dest='linear',
        help='Linear hidden size')

    # Logging configs
    argparser.add_argument(
        '--tensorboard',
        metavar='TB',
        default=True,
        help='Decides if to log information to tensorboard (default: False)')

    args = argparser.parse_known_args()
    if len(args) > 1:
        args = args[0]

    run_client(args)


def run_client(args):

    args.host = 'localhost'
    args.port = 2000
    args.linear = 256
    args.conv = 128

    client = configure_simulation(args)

    # Controller initialization
    if args.controller is 'MPC':
        TARGET_SPEED = args.speed
        STEPS_AHEAD = args.steps_ahead
        controller = MPCController(target_speed=TARGET_SPEED, steps_ahead=STEPS_AHEAD, dt=0.1)
    elif args.controller is 'NN':
        depth_shape = [3, 60, 80]
        actor_path = '/home/ksawi/Documents/Workspace/carla/carla_racetrack_BS/data/models/20200531_1407/DDPGActor_l128_conv32/train2.pt'
        critic_path = '/home/ksawi/Documents/Workspace/carla/carla_racetrack_BS/data/models/20200531_1315/DDPGCritic_l128_conv32/train.pt'
        actor_net = DDPGActor(img_shape=depth_shape, numeric_shape=[len(NUMERIC_FEATURES)],
                              output_shape=[2], linear_hidden=args.linear, conv_hidden=args.conv, cuda=False)
        actor_net.load_state_dict(torch.load(actor_path))
        critic_net = DDPGCritic(actor_out_shape=[2, ], depth_shape=depth_shape, numeric_shape=[len(NUMERIC_FEATURES)],
                            linear_hidden=args.linear, conv_hidden=args.conv, cuda=False)
        critic_net.load_state_dict(torch.load(critic_path))

        controller = NNController(actor_net=actor_net, critic_net=critic_net,
                                  features=NUMERIC_FEATURES,device='cpu')

    for i in range(args.episodes):
        status, save_path = run_episode(client=client, controller=controller, args=args)
        print(f'Episode {i+1} ended with status: {status}')
        print(f'Data saved in: {save_path}')



def run_episode(client:carla.Client, controller:Controller, args) -> (str, str):
    '''
    Runs single episode. Configures world and agent, spawns it on map and controlls it from start point to termination
    state.

    :param client: carla.Client, client object connected to the Carla Server
    :param actor: carla.Vehicle
    :param controller: inherits abstract Controller class
    :param spawn_points: orginal or inverted list of spawnpoints
    :param writer: SummaryWriter, logger for tensorboard
    :param viz: visdom.Vis, other logger #refactor to one dictionary
    :param args: argparse.args, config #refactor to dict
    :return: status:str, succes
             save_path
    '''
    NUM_STEPS = args.num_steps

    spawn_points_df = pd.read_csv(f'{DATA_PATH}/spawn_points/{args.map}.csv')
    spawn_points = df_to_spawn_points(spawn_points_df, n=10000, invert=args.inverse)

    environment = Environment(client=client)
    world = environment.reset_env(args)
    agent_config = {'world':world, 'controller':controller, 'vehicle':VEHICLES[args.vehicle],
                    'sensors':SENSORS, 'spawn_points':spawn_points}
    agent = Agent(**agent_config)
    agent.initialize_vehicle()
    spectator = world.get_spectator()
    spectator.set_transform(numpy_to_transform(
        spawn_points[agent.spawn_point_idx-30]))

    agent_transform = None
    world.tick()
    world.tick()
    # Calculate norm of all cordinates
    while (agent_transform != agent.transform).any():
        agent_transform = agent.transform
        world.tick()

    #INITIALIZE SENSORS
    agent.initialize_sensors()

    # Initialize visdom windows

    # Release handbrake
    for i in range(DATA_POINTS):
        world.tick()

    agent.init_reporting()
    agent._release_control()

    print('Control released')

    status = 'Max steps exceeded'
    for step in range(NUM_STEPS):
        #Retrieve state and actions

        state = agent.get_state(step, retrieve_data=True)

        #Apply action
        action = agent.play_step(state)

        #Transit to next state
        world.tick()
        next_state = {
            'velocity': agent.velocity,
            'location': agent.location
        }

        #Receive reward
        reward = environment.calc_reward(points_3D=agent.waypoints, state=state, next_state=next_state,
                                         gamma=GAMMA, step=step)

        if state['distance_2finish'] < 5:
            status = 'Finished'
            print(f'agent {str(agent)} finished the race in {step} steps')
            save_info(path=agent.save_path, state=state, action=action, reward=0)
            break

        if state['collisions'] > 0:
            status = 'Collision'
            print(f'failed, collision {str(agent)}')
            print(state['collisions'])
            time.sleep(3)
            save_info(path=agent.save_path, state=state, action=action,
                      reward=NEGATIVE_REWARD * (GAMMA ** step))
            agent.destroy()
            break

        save_info(path=agent.save_path, state=state, action=action, reward=reward)

        #Log
        if ((agent.velocity < 20) & (step % 10 == 0)) or (step % 100 == 0):
            set_spectator_above_actor(spectator, agent.transform)
        # time.sleep(0.1)

    #Calc Qvalues and add to reporting file
    update_Qvals(path=agent.save_path)

    world.tick()

    return status, agent.save_path


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')