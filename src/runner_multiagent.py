import json
import time
import argparse
import numpy as np
import pandas as pd
import carla

#Local imports
import torch
import visdom as vis

from control.nn_control import NNController
from environment import Agent, Environment
from net.ddpg_net import DDPGActor, DDPGCritic
from spawn import df_to_spawn_points, numpy_to_transform, set_spectator_above_actor, configure_simulation
from control.mpc_control import MPCController
from control.abstract_control import Controller
from tensorboardX import SummaryWriter

#Configs
from config import DATA_PATH, FRAMERATE, TENSORBOARD_DATA, GAMMA, \
    DATE_TIME, SENSORS, VEHICLES, CARLA_IP, MAP, INVERSE, NO_AGENTS, NEGATIVE_REWARD, DATA_POINTS, NUMERIC_FEATURES

from utils import tensorboard_log, visdom_log, visdom_initialize_windows, init_reporting, save_info, update_Qvals


#Use this script only for data generation
# for map in 'circut_spa' 'RaceTrack' 'RaceTrack2'; do for car in 0 1 2; do for speed in 150 100 80; do python runner_multiagent.py --map=$map --vehicle=$car --speed=$speed ; done; done; done;
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
        '--inverse',
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

    argparser.add_argument(
        '--no_agents',
        default=NO_AGENTS,
        type=int,
        dest='no_agents',
        help='no of spawned agents')

    # Logging configs
    argparser.add_argument(
        '--tensorboard',
        metavar='TB',
        default=True,
        help='Decides if to log information to tensorboard (default: False)')

    args = argparser.parse_known_args()
    if len(args) > 1:
        args = args[0]
    # return args
    run_client(args)


def run_client(args):

    args.host = 'localhost'
    args.port = 2000
    # args.controller = 'NN'

    client = configure_simulation(args)

    # Controller initialization - we initialize one controller for n-agents, what happens in multiprocessing.
    if args.controller is 'MPC':
        TARGET_SPEED = args.speed
        STEPS_AHEAD = args.steps_ahead
        controller = MPCController(target_speed=TARGET_SPEED, steps_ahead=STEPS_AHEAD, dt=0.1)
    elif args.controller is 'NN':
        depth_shape = [3, 60, 80]
        # args.linear = 64
        # args.conv = 64
        actor_path = '/home/ksawi/Documents/Workspace/carla/carla_racetrack_BS/data/models/20200604_2349/DDPGActor_l64_conv64/test/test.pt'
        critic_path = '/home/ksawi/Documents/Workspace/carla/carla_racetrack_BS/data/models/20200604_2318/DDPGCritic_l64_conv64/test/test.pt'
        actor_net = DDPGActor(img_shape=depth_shape, numeric_shape=[len(NUMERIC_FEATURES)],
                              output_shape=[2], linear_hidden=args.linear, conv_hidden=args.conv, cuda=False)
        actor_net.load_state_dict(torch.load(actor_path))
        critic_net = DDPGCritic(actor_out_shape=[2, ], img_shape=depth_shape, numeric_shape=[len(NUMERIC_FEATURES)],
                            linear_hidden=args.linear, conv_hidden=args.conv, cuda=False)
        critic_net.load_state_dict(torch.load(critic_path))

        controller = NNController(actor_net=actor_net, critic_net=critic_net, no_data_points=1,
                                  features=NUMERIC_FEATURES,device='cpu')

    for i in range(args.episodes):
        status, save_paths = run_episode(client=client,
                                        controller=controller,
                                        args=args)
        for (actor, status), path in zip(status.items(), save_paths):
            print(f'Episode {i + 1} actor {actor} ended with status: {status}')
            print(f'Data saved in: {path}')



def run_episode(client:carla.Client, controller:Controller, args) -> (dict, dict):
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
             actor_dict -> speed, wheels turn, throttle, reward -> can be taken from actor?
             env_dict -> consecutive locations of actor, distances to closest spawn point, starting spawn point
             array[np.array] -> photos
    '''
    NUM_STEPS = args.num_steps
    spawn_points_df = pd.read_csv(f'{DATA_PATH}/spawn_points/{args.map}.csv')
    spawn_points = df_to_spawn_points(spawn_points_df, n=10000, inverse=args.inverse)
    environment = Environment(client=client)
    world = environment.reset_env(args)
    agent_config = {'world':world, 'controller':controller, 'vehicle':VEHICLES[args.vehicle],
                    'sensors':SENSORS, 'spawn_points':spawn_points}
    environment.init_agents(no_agents=args.no_agents, agent_config=agent_config)
    spectator = world.get_spectator()
    spectator.set_transform(numpy_to_transform(
        spawn_points[environment.agents[0].spawn_point_idx-30]))

    environment.stabilize_vehicles()

    #INITIALIZE SENSORS
    environment.initialize_agents_sensors()

    for i in range(DATA_POINTS):
        world.tick()

    environment.initialize_agents_reporting()
    for agent in environment.agents:
        agent._release_control()
        print(f'{agent} control released')
    save_paths = [agent.save_path for agent in environment.agents]
    #TODO dump agent dict as json

    status = dict({str(agent): 'Max steps exceeded' for agent in environment.agents})

    for step in range(NUM_STEPS):

        states = [agent.get_state(step, retrieve_data=True) for agent in environment.agents]
        actions = [agent.play_step(state) for agent, state in zip(environment.agents, states)]
        world.tick()

        next_states = [{'velocity': agent.velocity,'location': agent.location} for agent in environment.agents]

        rewards = [environment.calc_reward(points_3D=agent.waypoints, state=state, next_state=next_state,
                                           gamma=GAMMA, step=step) for agent, state, next_state in zip(environment.agents, states, next_states)]

        for idx, (state, agent) in enumerate(zip(states, environment.agents)):
            if state['distance_2finish'] < 5:
                print(f'agent {str(agent)} finished the race in {step} steps car {args.vehicle}')
                save_info(path=agent.save_path, state=state, action=action, reward=0)
                status[str(agent)] = 'Finished'
                agent.destroy(data=True, step=step)
                environment.agents.pop(idx)

            if state['collisions'] > 0:
                print(f'failed, collision {str(agent)} at step {step}, car {args.vehicle}')
                save_info(path=agent.save_path, state=state, action=action,
                          reward=reward + NEGATIVE_REWARD * (GAMMA ** step))
                status[str(agent)] = 'Collision'
                agent.destroy(data=True, step=step)
                environment.agents.pop(idx)

        for agent, state, action, reward in zip(environment.agents, states, actions, rewards):
            save_info(path=agent.save_path, state=state, action=action, reward=reward)

        if len(environment.agents) < 1:
            print('fini')
            break

    if len(environment.agents) > 1:
        for agent in environment.agents:
            agent.destroy(data=True, step=NUM_STEPS)

    for agent_path in save_paths:
        update_Qvals(path=agent_path)

    world.tick()
    world.tick()

    return status, save_paths


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')