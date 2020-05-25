import time
import argparse
import numpy as np
import pandas as pd
import carla

#Local imports
import visdom as vis

from environment import Agent, Environment
from spawn import df_to_spawn_points, numpy_to_transform, set_spectator_above_actor, configure_simulation
from control.mpc_control import MPCController
from control.abstract_control import Controller
from tensorboardX import SummaryWriter

#Configs
from config import DATA_PATH, FRAMERATE, TENSORBOARD_DATA, GAMMA, \
    DATE_TIME, SENSORS, VEHICLE, CARLA_IP, MAP, INVERSE, NO_AGENTS, NEGATIVE_REWARD

from utils import tensorboard_log, visdom_log, visdom_initialize_windows, init_reporting, save_info, update_Qvals


#Use this script only for data generation

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
    # args = main()
    args.host = 'localhost'
    args.port = 2000

    args.tensorboard = False
    writer = None
    if args.controller == 'MPC':
        TARGET_SPEED = args.speed
        STEPS_AHEAD = args.steps_ahead
        if args.tensorboard:
            writer = SummaryWriter(f'{TENSORBOARD_DATA}/{args.controller}/{args.map}_TS{TARGET_SPEED}_H{STEPS_AHEAD}_FRAMES{args.frames}_{DATE_TIME}',
                                   flush_secs=5, max_queue=5)
    elif args.tensorboard:
        writer = SummaryWriter(f'{TENSORBOARD_DATA}/{args.controller}/{args.map}_FRAMES{args.frames}', flush_secs=5)

    # Connecting to client -> later package it in function which checks if the world is already loaded and if the settings are the same.
    # In order to host more scripts concurrently
    client = configure_simulation(args)

    # load spawnpoints from csv -> generate spawn points from notebooks/20200414_setting_points.ipynb
    spawn_points_df = pd.read_csv(f'{DATA_PATH}/spawn_points/{args.map}.csv')
    spawn_points = df_to_spawn_points(spawn_points_df, n=10000, inverse=INVERSE) #We keep it here in order to have one way simulation within one script

    # Controller initialization - we initialize one controller for n-agents, what happens in multiprocessing.
    if args.controller is 'MPC':
        controller = MPCController(target_speed=TARGET_SPEED, steps_ahead=STEPS_AHEAD, dt=0.1)

    status, actor_dict, env_dict, sensor_data = run_episode(client=client,
                                                            controller=controller,
                                                            spawn_points=spawn_points,
                                                            writer=writer,
                                                            args=args)



def run_episode(client:carla.Client, controller:Controller, spawn_points:np.array,
                writer:SummaryWriter, args) -> (str, dict, dict, list):
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

    environment = Environment(client=client)
    world = environment.reset_env(args)
    agent_config = {'world':world, 'controller':controller, 'vehicle':args.vehicle,
                    'sensors':SENSORS, 'spawn_points':spawn_points}
    environment.init_agents(no_agents=args.no_agents, agent_config=agent_config)
    environment.init_vehicles()
    spectator = world.get_spectator()
    spectator.set_transform(numpy_to_transform(
        spawn_points[environment.agents[0].spawn_point_idx-30]))

    environment.stabilize_vehicles()

    #INITIALIZE SENSORS
    environment.initialize_agents_sensors()

    # Release handbrake
    world.tick()
    time.sleep(1)
    environment.initialize_agents_reporting(sensors=SENSORS)
    save_paths = [agent.save_path for agent in environment.agents]

    for step in range(NUM_STEPS):  #TODO change to while with conditions
        #Retrieve state and actions

        states = [agent.get_state(step, retrieve_data=True) for agent in environment.agents]

        #Apply action
        actions = [agent.play_step(state) for agent, state in zip(environment.agents, states)]
        # actions.append(action)

        #Transit to next state
        world.tick()

        next_states = [{'velocity': agent.velocity,'location': agent.location} for agent in environment.agents]

        rewards = [environment.calc_reward(points_3D=agent.waypoints, state=state, next_state=next_state,
                                           gamma=GAMMA, step=step) for agent, state, next_state in zip(environment.agents, states, next_states)]

        for idx, (state, agent) in enumerate(zip(states, environment.agents)):
            if state['distance_2finish'] < 5:
                print(f'agent {str(agent)} finished the race in {step} steps')
                save_info(path=agent.save_path, state=state, action=action, reward=0)
                agent.destroy(data=True)
                environment.agents.pop(idx)

            if state['collisions'] > 0:
                print(f'failed, collision {str(agent)}')
                save_info(path=agent.save_path, state=state, action=action,
                          reward=reward + NEGATIVE_REWARD * (GAMMA ** step))
                agent.destroy(data=True)
                environment.agents.pop(idx)

        for agent, state, action, reward in zip(environment.agents, states, actions, rewards):
            save_info(path=agent.save_path, state=state, action=action, reward=reward)

        if len(environment.agents) < 1:
            print('fini')
            break

    for agent_path in save_paths:
        update_Qvals(path=agent_path)

    status, actor_dict, env_dict, sensor_data = str, dict, dict, list

    return status, actor_dict, env_dict, sensor_data


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')