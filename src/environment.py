import argparse
import copy
import datetime
import json
import os

import numpy as np
import carla
import torch
import torch.multiprocessing as mp

#Local imports
from config import IMAGE_DOWNSIZE_FACTOR, FRAMERATE, DATA_PATH, DATE_TIME, SENSORS, INVERSE, DATA_POINTS
from control.abstract_control import Controller
from spawn import sensors_config, numpy_to_transform, velocity_to_kmh, transform_to_numpy, location_to_numpy, \
    to_vehicle_control, control_to_gas_brake
from utils import to_rgb, to_array, calc_distance, save_img, init_reporting


# For saving imgs
# https://github.com/drj11/pypng/

class Agent:
    def __init__(self, world:carla.World, controller:Controller, vehicle:str, sensors:dict, spawn_points:np.array, spawn_point_idx:int=None, no_data_points:int=DATA_POINTS):
        '''
        All of the default data is stored in the form of numpy array,
        transforms to other formats are performed ad hoc.
        :param world:carla.World
        :param controller:Controller, class inherited from abstract class Controller providing control method
        :param vehicle:str, exact vehicle blueprint name
        :param sensors:dictionary imported from config describing which sensors agent should use
        :param spawn_points: np.array
        :param spawn_point_idx:int,
        :param no_data_points:int,
        '''
        self.world = world
        self.map = world.get_map().name
        self.actor:carla.Vehicle = self.world.get_blueprint_library().find(vehicle)
        self.date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.controller = controller
        self.sensors = sensors_config(self.world.get_blueprint_library(), **sensors)
        self.spawn_point_idx = spawn_point_idx or int(np.random.randint(len(spawn_points)))
        self.spawn_point = spawn_points[self.spawn_point_idx]
        self.waypoints = np.concatenate([spawn_points[self.spawn_point_idx:, :], spawn_points[:self.spawn_point_idx-20, :]])[:, :3]  # delete yaw column, does not allow agent to go backward
        self.initialized = False
        self.sensors_initialized = False
        self.no_data_points = no_data_points

    def __str__(self) -> str:
        return f'{self.controller.__class__.__name__}_{"_".join(self.sensors.keys())}_{self.spawn_point_idx}'

    def __dict__(self) -> dict:
        agent = {'name': str(self),
                 'map': self.map,
                 'save_path': self.save_path,
                 'spawn_point_idx': self.spawn_point_idx,
                 'no_data_points': self.no_data_points,
                 'sensors': self.sensors.keys(),
                 'controller': self.controller.name,
                 'vehicle': self.actor.id
                 }
        return agent

    @property
    def save_path(self) -> str:
        return f'{DATA_PATH}/experiments/{self.map}{"_inverse"*INVERSE}/{self.date_time}/{self.__str__()}'

    @property
    def transform(self):
        return transform_to_numpy(self.actor.get_transform())

    @property
    def location(self) -> np.array:
        return location_to_numpy(self.actor.get_location())

    @property
    def velocity(self) -> float:
        return velocity_to_kmh(self.actor.get_velocity())

    @property
    def velocity_vec(self) -> carla.Vector3D:
        # velocity has the same vector structure as location
        return location_to_numpy(self.actor.get_velocity())


    def play_step(self, state:dict, batch:bool=False) -> dict:
        '''
        Plays one step of simulation, chooses action to take in a given state
        :param state:dict, dictionary of all nescessary values observable by actor provided by the state
        :param batch:bool, apply the action or return it as a value (for asynchronous methods)
        :return: action:dict, dictionary containing values for actuators
        '''
        action = self.controller.control(
            state=state,
            pts_3D=self.waypoints
        )

        if not batch:
            self.actor.apply_control(
                to_vehicle_control(
                    gas_brake=action['gas_brake'],
                    steer=action['steer']
                ))

        return action

    #TODO add initial empty state from which keys will be extracted for logging

    def get_state(self, step, retrieve_data:bool=False, **kwargs):
        '''
        Retrieves information about the state from agent's sensors
        :param step:int, step number needed for computation of indexes for logging
        :param retrieve_data:bool, wether 
        :return:
        '''
        state = dict({})
        state['step'] = step

        sensors = list(self.sensors.keys())
        sensors.remove('collisions')

        for sensor in sensors:
            indexes = self.get_sensor_data_indexes(step)
            state[f'{sensor}_indexes'] = indexes
            if retrieve_data:
                data = self.get_sensor_data(sensor)
                state[f'{sensor}_data'] = data
                self._release_data(sensor=sensor, step=state['step'])

        collision = sum(self.sensors['collisions']['data'])
        state['collisions'] = collision if collision > 2_500 else 0
        control = self.actor.get_control()
        state['state_steer'] = control.steer
        state['state_gas_brake'] = control_to_gas_brake(control)
        state['velocity'] = self.velocity
        state['velocity_vec'] = list(self.velocity_vec)
        state['yaw'] = self.transform[3] #hardcoded thats bad
        state['location'] = list(self.location)
        state['distance_2finish'] = calc_distance(actor_location=state['location'],
                                                  points_3D=self.waypoints)




        return state

    def get_sensor_data(self, sensor) -> list:
        '''
        Returns particular sensor data from current agent state.
        :param sensor:
        :return: list of datapoints
        '''
        # if len(self.sensors[sensor]['data']) < self.no_data_points:
        #     data = [self.sensors[sensor]['data'][0] for i in
        #             range(self.no_data_points - len(self.sensors[sensor]['data']))] \
        #            + [frame for frame in self.sensors[sensor]['data']]
        # else:
        data = self.sensors[sensor]['data'][-self.no_data_points:]
        return data

    def get_sensor_data_indexes(self, step) -> list:
        '''
        Returns global indexes of data associated with particular step.
        :param step: int, current step
        :return: list of integers
        '''
        # if step < self.no_data_points:
        #     indexes = [0 for i in range(self.no_data_points - step)] \
        #            + [idx + 1 for idx in range(step)]
        # else:
        indexes = [idx for idx in range(step, step + self.no_data_points)]
        return indexes

    def set_waypoints(self, spawn_points:np.array, spawn_point_idx:int):
        '''
        For explicit selection of spawn point and waypoints of agent
        :param spawn_points: np.array, consecutive points forming race track (x,y,z,yaw)
        :param spawn_point_idx: int
        :return: None
        '''
        self.spawn_point_idx = spawn_point_idx
        self.spawn_point = spawn_points[spawn_point_idx]
        self.waypoints = np.concatenate(
            [spawn_points[self.spawn_point_idx:, :],
             spawn_points[:self.spawn_point_idx, :]]
        )[:,:3]  # delete yaw column

    def initialize_vehicle(self) -> None:
        '''
        Spawns vehicle and applies break
        :return: None
        '''
        if not self.initialized:
            try:
                self.actor:carla.Vehicle = self.world.spawn_actor(self.actor, numpy_to_transform(self.spawn_point))
                self.actor.apply_control(carla.VehicleControl(brake=1., gear=1))
                self.initialized = True
                print('Vehicle initilized')
            except:
                raise Exception('Vehicle not spawned')
        else:
            raise Exception('Vehicle already spawned')

    def initialize_sensors(self) -> None:
        '''
        Initializes sensors based on intial sensor dict loaded from config
        :return: None
        '''
        if 'depth' in self.sensors.keys():
            self.sensors['depth']['data'] = []
            self.sensors['depth']['actor'] = self.world.spawn_actor(blueprint=self.sensors['depth']['blueprint'],
                                                  transform=self.sensors['depth']['transform'],
                                                  attach_to=self.actor)
            self.sensors['depth']['actor'].listen(lambda img_raw: (img_raw.convert(self.sensors['depth']['color_converter']), \
                                                 self.sensors['depth']['data'].append(to_rgb(to_array(img_raw)))))

        if 'rgb' in self.sensors.keys():
            self.sensors['rgb']['data'] = []
            self.sensors['rgb']['actor'] = self.world.spawn_actor(
                blueprint=self.sensors['rgb']['blueprint'],
                transform=self.sensors['rgb']['transform'],
                attach_to=self.actor
            )
            self.sensors['rgb']['actor'].listen(lambda img_raw: self.sensors['rgb']['data'].append(to_rgb(to_array(img_raw))))

        if 'segmentation' in self.sensors.keys():
            self.sensors['segmentation']['data'] = []
            self.sensors['segmentation']['actor'] = self.world.spawn_actor(blueprint=self.sensors['segmentation']['blueprint'],
                                                  transform=self.sensors['segmentation']['transform'],
                                                  attach_to=self.actor)
            self.sensors['segmentation']['actor'].listen(lambda img_raw: (img_raw.convert(self.sensors['segmentation']['color_converter']), \
                                                 self.sensors['segmentation']['data'].append(to_rgb(to_array(img_raw)))))


        if 'collisions' in self.sensors.keys():
            self.sensors['collisions']['data'] = [0]
            self.sensors['collisions']['actor'] = self.world.spawn_actor(
                blueprint=self.sensors['collisions']['blueprint'],
                transform=self.sensors['collisions']['transform'],
                attach_to=self.actor
            )
            self.sensors['collisions']['actor'].listen(lambda collision: \
                self.sensors['collisions']['data'].insert(0, sum(location_to_numpy(collision.normal_impulse))))

        self.sensors_initialized = True
        print('Sensors initialized')
        # self._release_control()
        # print('Control released')

    def _release_control(self) -> None:
        '''
        Private method releasing control of vahicle before the start of simulation.
        :return: None
        '''
        self.actor.apply_control(carla.VehicleControl(throttle=0., brake=0., gear=1))

    def _release_data(self, sensor:str, step: int) -> None:
        '''
        Private method which saves sensor data to the disk and releases memory.
        :param step:
        :return: None
        '''
        file = f'{sensor}_{step}.png'
        save_img(img=self.sensors[sensor]['data'][-1], path=f'{self.save_path}/sensors/{file}')
        # if step > self.no_data_points:
        self.sensors[sensor]['data'].pop(0)
            #For future buffer change save to bulk save of all data from sensors till [-self.data_points:]
            # self.sensors[sensor]['data'].pop(0)

    def destroy(self, data:bool=False) -> None:
        '''
        Destroying agent entities while preserving sensors data.
        :param: data:bool, decides of cleaning data asociated with agent from buffer.
        :return:bool, if Agent destroyed.
        '''
        if self.sensors_initialized:
            for sensor in self.sensors:
                self.sensors[sensor]['actor'].destroy()
        self.actor.destroy()

        if data:
            self.sensors = None

        return True

    def init_reporting(self) -> None:
        '''
        Initialize file for logging based on suite of utilized sensors
        :param path:str, path to the experiment folder
        :param sensors: dict of sensors from config consisting boolean values
        :return: None
        '''

        state = self.get_state(step=0, retrieve_data=False)

        # Apply action
        action = self.play_step(state, batch=True)

        if len(self.save_path.split('/')) > 1:
            os.makedirs(name='/'.join(self.save_path.split('/')), exist_ok=True)

        keys = list({**state, **action}.keys())
        header = ','.join([f'{k}' for k in keys])
        header = f'{header},reward\n'

        with open(f'{self.save_path}/episode_info.csv', 'w+') as file:
            file.write(header)

        json.dump(dict(self), open(f'{self.save_path}/agent_info.json', 'w+'))

        print('Init succesfull')

class Environment:
    #TODO implement as Singleton
    #TODO implement multiagent handling methods for multiprocessing:
    # initialization, state-action method, rewards calculation method, logging (Global Summary Writer).
    def __init__(self, client:carla.Client):
        '''
        Orchestrates asynchronous agents and world ticks.
        Calculates reward and controls the state of the world.

        :param client: carla.Client
        '''
        self.client = client
        self.world = None
        self.agents = []


    def reset_env(self, args:argparse.ArgumentParser) -> carla.World:
        '''
        Loads map provided with args
        #TODO change args to map, synchronous and frame parameters.
        :param args:
        :return:
        '''

        if self.agents:
            for agent in self.agents:
                agent.destroy(data=True)
            self.agents = []

        if self.client.get_world().get_map().name != args.map:
            self.world: carla.World = self.client.load_world(args.map)
        else:
            self.world:carla.World = self.client.reload_world()

        if args.synchronous & (not self.world.get_settings().synchronous_mode):
            settings = self.world.get_settings()
            settings.synchronous_mode = True  # Enables synchronous mode
            settings.fixed_delta_seconds = 1 / args.frames
            self.world.apply_settings(settings)

        return self.world

    def init_agents(self, no_agents:int, agent_config:dict) -> None:
        '''

        :param no_agents:
        :param agent_config:
        :return:
        '''
        points_len = len(agent_config['spawn_points'])
        spawn_point_indexes = (np.linspace(0, points_len - (points_len/no_agents), no_agents, dtype=np.int) + \
                               np.random.randint(0, points_len)) % points_len
        for idx in spawn_point_indexes:
                current_agent_config = {**agent_config, 'spawn_point_idx':idx}
                agent = Agent(**current_agent_config)
                try:
                    agent.initialize_vehicle()
                    self.agents.append(agent)
                except Exception as e:
                    print(e)
        self.world.tick()
        self.world.tick()


    def stabilize_vehicles(self) -> None:
        '''
        Stabilizes vehicles on their start points
        :return:
        '''

        def eq_transforms(agents_transforms:np.array, current_agents_transforms:np.array) -> bool:
            '''
            Compares previous agents transforms with current ones
            :param agents_transforms: np.array
            :param current_agents_transforms: np.array
            :return: bool
            '''
            value = np.array([(prev==curr).all() for prev, curr in zip(agents_transforms, current_agents_transforms)]).all()
            return value

        at = np.array([None for agent in self.agents])
        no_ticks = 0
        while True:
            cat = np.array([agent.transform for agent in self.agents])
            if eq_transforms(agents_transforms=at, current_agents_transforms=cat) or no_ticks > 100:
                break
            at = cat
            self.world.tick()
            no_ticks += 1

    def initialize_agents_sensors(self) -> None:
        '''
        Initilizes sensors for every agent
        :return: None
        '''
        for agent in self.agents:
            agent.initialize_sensors()

    def initialize_agents_reporting(self) -> None:
        '''
        Initializes reporting files for every agent.
        :return: None
        '''
        for agent in self.agents:
            agent.init_reporting()

    def get_agents_states(self, step:int, retrieve_data:bool=False) -> list:
        '''
        Multiprocessing state retrieval from agents
        :param step:
        :param retrieve_data:
        :return:
        '''

        # https://stackoverflow.com/questions/29630217/multiprocessing-in-ipython-console-on-windows-machine-if-name-requirement
        # https://stackoverflow.com/questions/23665414/multiprocessing-using-imported-modules
        # https://pymotw.com/2/multiprocessing/basics.html -> subclassing process i importable target function

        #TODO wrong
        # TODO https://docs.python.org/3/library/concurrency.html
        #TODO https://docs.python.org/3/library/multiprocessing.html - read all

        def worker(agent, step, retrieve_data, states):
            state = agent.get_state(step, retrieve_data)
            states[str(agent)] = state

        states = dict({})
        processes = []
        for agent in self.agents:
            p = mp.Process(target=worker, args=(agent, step, retrieve_data, states))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return copy.deepcopy(states)

    def get_agents_actions(self, states:list) -> list:
        pass

    def get_agents_states_actions(self, step:int, retrieve_data:bool=False) -> dict:
        pass

    def destroy_agents(self):
        for agent in self.agents:
            agent.destroy(data=True)
        self.agents = []

    def toggle_world(self, frames:int=FRAMERATE) -> None:
        '''
        Toggle world state from synchonous mode to normal mode.
        For debugging purposes.
        :param frames: int, number of simulated frames per second
        :return: None
        '''
        settings = self.world.get_settings()
        settings.synchronous_mode = not settings.synchronous_mode
        settings.fixed_delta_seconds = abs(float(settings.fixed_delta_seconds or 0) - 1/frames)
        self.world.apply_settings(settings)

    def calc_reward(self, points_3D:np.array, state:dict, next_state, gamma: float = .995, punishment:float=0.05, step: int = 0) -> float:
        '''
        Calculating reward based on location and speed between 2 consecutive states.

        :param points_3D:np.array, points
        :param state:
        :param next_state:dict, simple dict containing only velocity and location from resulting state
        :param gamma:float, discount factor
        :param step:int
        :return: reward:float,
        '''

        if calc_distance(actor_location=next_state['location'], points_3D=points_3D) >= calc_distance(
                actor_location=state['location'], points_3D=points_3D):
            return -(next_state['velocity'] / (state['velocity'] + 0.2)) * (gamma ** step) - punishment

        return (next_state['velocity'] / (state['velocity'] + 0.2)) * (gamma ** step) - punishment

    def calc_reward_distance(self, points_3D:np.array, state:dict, next_state, gamma: float = .995, punishment:float=0.05, step: int = 0) -> float:
        state_distance = calc_distance(actor_location=state['location'], points_3D=points_3D)
        next_distance = calc_distance(actor_location=next_state['location'], points_3D=points_3D)

        reward = (((state_distance - 1e-50) / next_distance) - 1) * 100 *  (gamma ** step) - punishment

        return reward