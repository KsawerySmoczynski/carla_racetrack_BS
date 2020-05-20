import argparse
import numpy as np
import carla


from config import IMAGE_DOWNSIZE_FACTOR, FRAMERATE, DATA_PATH, DATE_TIME
from control.abstract_control import Controller

#NAGRODA (Dystans do - Dystans przejechany)
from spawn import sensors_config, numpy_to_transform, velocity_to_kmh, transform_to_numpy, location_to_numpy, \
    to_vehicle_control
from utils import to_rgb_resized, to_array, calc_distance, save_img


# For saving imgs
# https://github.com/drj11/pypng/

class Agent:
    def __init__(self, world:carla.World, controller:Controller, vehicle:str, sensors:dict, spawn_points:np.array, no_data_points:int=4):
        '''
        All of the default data is stored in the form of numpy array,
        transforms to other formats are performed ad hoc.
        :param world:carla.World
        :param controller:Controller, class inherited from abstract class Controller providing control method
        :param vehicle:str, exact vehicle blueprint name
        :param sensors:dictionary imported from config describing which sensors agent should use
        '''
        self.world = world
        self.map = world.get_map().name
        self.actor:carla.Vehicle = self.world.get_blueprint_library().find(vehicle)
        self.controller = controller
        self.sensors = sensors_config(self.world.get_blueprint_library(), depth=sensors['depth'],
                                      collision=sensors['collisions'], rgb=sensors['rgb'])
        self.spawn_point_idx = int(np.random.randint(len(spawn_points)))
        self.spawn_point = spawn_points[self.spawn_point_idx]
        self.waypoints = np.concatenate([spawn_points[self.spawn_point_idx:, :], spawn_points[:self.spawn_point_idx, :]])[:, :3]  # delete yaw column
        self.initialized = False
        self.sensors_initialized = False
        self.no_data_points = no_data_points

    def __str__(self) -> str:
        return f'{self.controller.__class__.__name__}_{"_".join(self.sensors.keys())}_{self.spawn_point_idx}'

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
        return self.actor.get_velocity()


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

    def get_state(self, step, retrieve_data:bool=False):
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
                state[sensor] = data

        state['collisions'] = sum(self.sensors['collisions']['data'])
        state['velocity'] = self.velocity
        state['velocity_vec'] = self.velocity_vec
        state['yaw'] = self.transform[3] #hardcoded thats bad
        state['location'] = self.location
        state['distance_2finish'] = calc_distance(actor_location=state['location'],
                                                  points_3D=self.waypoints)
        self._release_data(state['step'])

        return state

    def get_sensor_data(self, sensor) -> list:
        '''
        Returns particular sensor data from current agent state.
        :param sensor:
        :return: list of datapoints
        '''
        if len(self.sensors[sensor]['data']) < self.no_data_points:
            data = [self.sensors[sensor]['data'][0] for i in
                    range(self.no_data_points - len(self.sensors[sensor]['data']))] \
                   + [frame for frame in self.sensors[sensor]['data']]
        else:
            data = self.sensors[sensor]['data'][-self.no_data_points:]
        return data

    def get_sensor_data_indexes(self, step) -> list:
        '''
        Returns global indexes of data associated with particular step.
        :param step: int, current step
        :return: list of integers
        '''
        if step < self.no_data_points:
            indexes = [0 for i in range(self.no_data_points - step)] \
                   + [idx + 1 for idx in range(step)]
        else:
            indexes = [idx for idx in range(step-self.no_data_points, step)]
        return indexes


    def initialize_vehicle(self) -> None:
        '''
        Spawns vehicle and applies break
        :return: None
        '''
        if not self.initialized:
            self.actor:carla.Vehicle = self.world.spawn_actor(self.actor, numpy_to_transform(self.spawn_point))
            self.actor.apply_control(carla.VehicleControl(brake=1., gear=1))
            self.initialized = True
            print('Vehicle initilized')
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
                                                 self.sensors['depth']['data'].append(to_rgb_resized(to_array(img_raw)))))

        if 'collisions' in self.sensors.keys():
            self.sensors['collisions']['data'] = []
            self.sensors['collisions']['actor'] = self.world.spawn_actor(
                blueprint=self.sensors['collisions']['blueprint'],
                transform=self.sensors['collisions']['transform'],
                attach_to=self.actor
            )
            self.sensors['collisions']['actor'].listen(lambda collision: \
                                                           self.sensors['collisions']['data'].append(isinstance(collision.normal_impulse, carla.Vector3D)))


        if 'rgb' in self.sensors.keys():
            self.sensors['rgb']['data'] = []
            self.sensors['rgb']['actor'] = self.world.spawn_actor(
                blueprint=self.sensors['rgb']['blueprint'],
                transform=self.sensors['rgb']['transform'],
                attach_to=self.actor
            )
            self.sensors['rgb']['actor'].listen(lambda img_raw: self.sensors['rgb']['data'].append(to_rgb_resized(to_array(img_raw))))

        self.sensors_initialized = True
        print('Sensors initialized')
        self._release_control()
        print('Control released')

    def _release_control(self) -> None:
        '''
        Private method releasing control of vahicle before the start of simulation.
        :return: None
        '''
        self.actor.apply_control(carla.VehicleControl(throttle=0., brake=0., gear=1))

    def _release_data(self, step: int) -> None:
        '''
        Private method which saves sensor data to the disk and releases memory.
        :param step:
        :return: None
        '''
        for sensor in self.sensors.keys():
            if sensor is not 'collisions':
                save_path = f'{DATA_PATH}/experiments/{self.map}/{DATE_TIME}_{self.__str__()}/{sensor}_{step}.png'
                save_img(img=self.sensors[sensor]['data'][-1], path=save_path)
                if step > self.no_data_points:
                    self.sensors[sensor]['data'].pop(0)

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
        self.controller = None

        if data:
            self.sensors = None

        return True


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
        # self.agents = []


    def reset_env(self, args:argparse.ArgumentParser) -> carla.World:
        '''
        Loads map provided with args
        #TODO change args to map, synchronous and frame parameters.
        :param args:
        :return:
        '''
        # [agent.actor.destroy() for agent in self.agents]
        # self.agents = []
        self.world: carla.World = self.client.load_world(args.map)
        if args.synchronous:
            settings = self.world.get_settings()
            settings.synchronous_mode = True  # Enables synchronous mode
            settings.fixed_delta_seconds = 1 / args.frames
            self.world.apply_settings(settings)
        return self.world

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

    def calc_reward(self, points_3D:np.array, state:dict, next_state, alpha: float = .995, step: int = 0) -> float:
        '''
        Calculating reward based on location and speed between 2 consecutive states.

        :param points_3D:np.array, points
        :param state:
        :param next_state:dict, simple dict containing only velocity and location from resulting state
        :param alpha:float, discount factor
        :param step:int
        :return: reward:float,
        '''

        if calc_distance(actor_location=next_state['location'], points_3D=points_3D) > calc_distance(
                actor_location=state['location'], points_3D=points_3D):
            return - (next_state['velocity'] / (state['velocity'] + 1)) * (alpha ** step)

        # return (next_state['velocity']/(state['velocity'] + 1)) * \
        #        ((tracklen - calc_from_closest_distance) - distance_travelled )  * (alpha**step)
        return (next_state['velocity'] / (state['velocity'] + 1)) * (alpha ** step)