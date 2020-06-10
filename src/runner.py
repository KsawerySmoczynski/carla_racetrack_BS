import json
import time
import argparse
import numpy as np
import pandas as pd
import carla

#Local imports
import torch
from control.nn_control import NNController
from environment import Environment
from net.ddpg_net import DDPGActor, DDPGCritic
from net.utils import ReplayBuffer
from spawn import df_to_spawn_points, numpy_to_transform, configure_simulation
from control.mpc_control import MPCController
from control.abstract_control import Controller


#Configs
from config import DATA_PATH, FRAMERATE, GAMMA, SENSORS, VEHICLES, \
    CARLA_IP, MAP, NO_AGENTS, NEGATIVE_REWARD, DATA_POINTS, NUMERIC_FEATURES, FEATURES_FOR_BATCH, BATCH_SIZE

from utils import save_info, update_Qvals, arg_bool, save_terminal_state


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
        '--invert',
        default='False',
        type=str,
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
        type=str,
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
    args.invert = arg_bool(args.invert)

    client = configure_simulation(args)

    # Controller initialization - we initialize one controller for n-agents, what happens in multiprocessing.
    if args.controller == 'MPC':
        TARGET_SPEED = args.speed
        STEPS_AHEAD = args.steps_ahead
        controller = MPCController(target_speed=TARGET_SPEED, steps_ahead=STEPS_AHEAD, dt=0.1)
    elif args.controller == 'NN':
        depth_shape = [3, 60, 80]

        actor_path = '/home/ksawi/Documents/Workspace/carla/carla_racetrack_BS/data/models/20200607_0017/DDPGActor_l64_conv64/test/test_10.pt'
        critic_path = '/home/ksawi/Documents/Workspace/carla/carla_racetrack_BS/data/models/20200606_2208/DDPGCritic_l64_conv64/test/test_13.pt'
        actor_net = DDPGActor(img_shape=depth_shape, numeric_shape=[len(NUMERIC_FEATURES)],
                              output_shape=[2], linear_hidden=args.linear, conv_hidden=args.conv, cuda=True)
        actor_net.load_state_dict(torch.load(actor_path))
        critic_net = DDPGCritic(actor_out_shape=[2, ], img_shape=depth_shape, numeric_shape=[len(NUMERIC_FEATURES)],
                            linear_hidden=args.linear, conv_hidden=args.conv, cuda=True)
        critic_net.load_state_dict(torch.load(critic_path))

        controller = NNController(actor_net=actor_net, critic_net=critic_net, no_data_points=1,
                                  features=NUMERIC_FEATURES, train=False, optimizer=None, device='cuda:0')
    else:
        print(args.controller)
        controller = None
        print('Unsuccesfull choice of controller, aborting')
        exit(1)

    #TODO
    # Initialize replay buffer
    # Remember currently loaded dfs
    transform = None #TODO define
    buffer = None
    # buffer = ReplayBuffer(capacity=100_000, features=FEATURES_FOR_BATCH, transform=transform,
    #                       batch_size=BATCH_SIZE, **SENSORS)
    # tag=None
    # samples = get_paths(as_tuples=True, shuffle=True, sensors=SENSORS, tag=None)
    # buffer._add_bulk(samples=samples)
    # prievous = []
    for i in range(args.episodes):
        #TODO
        # Compare remembered dfs with those in buffer -> set from buffer
        #       * existing ones remain
        #       * if df is in remembered but not in the buffer eliminate it from dfs
        #       * if df is not remembered but in the buffer delete it from the dfs and reload it from disk
        # buffer._load_dfs(prievous=prievous)
        # prievous = buffer.df_paths
        # print(buffer.pos)
        buffer, status, save_paths = run_episode(client=client,
                                        controller=controller,
                                        buffer=buffer,
                                        args=args)

        for (actor, status), path in zip(status.items(), save_paths):
            print(f'Episode {i + 1} actor {actor} ended with status: {status}')
            print(f'Data saved in: {path}')



def run_episode(client:carla.Client, controller:Controller, buffer:ReplayBuffer, args) -> (ReplayBuffer, dict, dict):
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
    spawn_points = df_to_spawn_points(spawn_points_df, n=10000, invert=args.invert)
    environment = Environment(client=client)
    world = environment.reset_env(args)
    agent_config = {'world':world, 'controller':controller, 'vehicle':VEHICLES[args.vehicle],
                    'sensors':SENSORS, 'spawn_points':spawn_points, 'invert':args.invert}
    environment.init_agents(no_agents=args.no_agents, agent_config=agent_config)
    if len(environment.agents) < 1:
        return buffer, dict({}), []
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
    status = dict({str(agent): 'Max steps exceeded' for agent in environment.agents})
    slow_frames = [0 for i in range(len(environment.agents))]

    for step in range(NUM_STEPS):

        states = [agent.get_state(step, retrieve_data=True) for agent in environment.agents]
        actions = [agent.play_step(state) for agent, state in zip(environment.agents, states)]

        world.tick()

        next_states = [{'velocity': agent.velocity,'location': agent.location} for agent in environment.agents]

        rewards = []
        for agent, state, next_state in zip(environment.agents, states, next_states):
            reward = environment.calc_reward(points_3D=agent.waypoints, state=state, next_state=next_state,
                                    gamma=GAMMA, step=step, punishment=NEGATIVE_REWARD/agent.initial_distance)
            rewards.append(reward)

        for idx, (state, action, reward, agent) in enumerate(zip(states, actions, rewards, environment.agents)):
            if agent.distance_2finish < 50:
                print(f'agent {str(agent)} finished the race in {step} steps car {args.vehicle}')
                #positive reward for win -> calculate it stupid
                step_info = save_info(path=agent.save_path, state=state, action=action, reward=reward)
                # buffer.add_step(path=agent.save_path, step=step_info)
                status[str(agent)] = 'Finished'
                terminal_state = agent.get_state(step=step+1, retrieve_data=True)
                save_terminal_state(path=agent.save_path, state=terminal_state, action=action)
                agent.destroy(data=True, step=step)
                environment.agents.pop(idx)
                continue

            elif agent.collision > 0:
                print(f'failed, collision {str(agent)} at step {step}, car {args.vehicle}')
                step_info = save_info(path=agent.save_path, state=state, action=action,
                          reward=reward - NEGATIVE_REWARD * (GAMMA ** step))
                # buffer.add_step(path=agent.save_path, step=step_info)
                status[str(agent)] = 'Collision'
                terminal_state = agent.get_state(step=step+1, retrieve_data=True)
                save_terminal_state(path=agent.save_path, state=terminal_state, action=action)
                agent.destroy(data=True, step=step)
                environment.agents.pop(idx)
                continue

            if state['velocity'] < 10:
                if slow_frames[idx] > 100:
                    print(f'agent {str(agent)} stuck, finish on step {step}, car {args.vehicle}')
                    # df = pd.read_csv(f'{agent.save_path}/episode_info.csv').iloc[:-70,:]
                    # Find max diff between consecutive speeds and cut there it will be your idx
                    # df.to_csv(f'{agent.save_path}/episode_info.csv', index=False)
                    # state['step'] = step-70
                    # step_info = save_info(path=agent.save_path, state=state, action=action,
                    #           reward=reward - NEGATIVE_REWARD * (GAMMA ** step))
                    step_info = save_info(path=agent.save_path, state=state, action=action,
                              reward=reward - NEGATIVE_REWARD * (GAMMA ** step))
                    # buffer.add_step(path=agent.save_path, step=step_info)
                    status[str(agent)] = 'Stuck'
                    terminal_state = agent.get_state(step=step+1, retrieve_data=True)
                    terminal_state['collisions'] = 2500
                    save_terminal_state(path=agent.save_path, state=terminal_state, action=action)
                    agent.destroy(data=True, step=step)
                    environment.agents.pop(idx)
                    continue
                slow_frames[idx] += 1

            step_info = save_info(path=agent.save_path, state=state, action=action, reward=reward)
                # step_info = save_info(path=agent.save_path, state=state, action=action, reward=reward)
                # buffer.add_step(path=agent.save_path, step=step_info)

        #TODO
        # * sample batch * no_agents
        # *
        # if args.controller == 'NN':
        #     batch = buffer.sample(batch_size=32)
        #     actor_loss_v, critic_loss_v, q_ref_v = controller.train_on_batch(batch=batch, gamma=GAMMA)
            #All lossess add to local tracker
            #Add losses for averaging to global tracker

        if len(environment.agents) < 1:
            print('fini')
            break

    if len(environment.agents) > 1:
        for agent in environment.agents:
            agent.destroy(data=True, step=NUM_STEPS)

    for (agent, info), path in zip(status.items(), save_paths):
        df = pd.read_csv(f'{path}/episode_info.csv')
        if args.controller == 'MPC':
            idx = 26
            df.loc[:idx, 'steer'] = 0.
            df.loc[:idx, 'state_steer'] = 0.
        if info == 'Max steps exceeded':
            idx = len(df)-1
            df.loc[idx-1,'reward'] = -NEGATIVE_REWARD * (GAMMA ** NUM_STEPS) #TODO -> discuss if necessary
            df.loc[idx,'steer'] = 0.
            df.loc[idx,'gas_brake'] = 0.
            df.loc[idx,'reward'] = 0. #TODO -> discuss if necessary
            df.loc[idx,'done'] = 1.
        #Update qvalues
        df['q'] = [sum(df['reward'][i:]) for i in range(df.shape[0])]
        df.to_csv(f'{path}/episode_info.csv', index=False)

    world.tick()
    world.tick()

    return buffer, status, save_paths


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')