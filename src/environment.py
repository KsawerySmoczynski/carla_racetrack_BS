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
    '''
    All of the default data is stored in the form of numpy array,
    transforms to other formats are performed ad hoc.
    '''
    def __init__(self, world:carla.World, controller:Controller, vehicle:str, sensors:dict, spawn_points:np.array):
        '''

        :param world:
        :param controller:
        :param vehicle:
        :param sensors:
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
        self.no_data_points = 4 #make setter for it, or parameter

    @property
    def transform(self):
        return transform_to_numpy(self.actor.get_transform())

    @property
    def location(self):
        return location_to_numpy(self.actor.get_location())

    @property
    def velocity(self):
        return velocity_to_kmh(self.actor.get_velocity())

    @property
    def velocity_vec(self):
        return self.actor.get_velocity()


    def play_step(self, state:dict, batch:bool=False) -> dict:
        #Sensors data has to be added
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
        state = dict({})
        state['step'] = step

        sensors = list(self.sensors.keys())
        sensors.remove('collisions')

        for sensor in sensors:
            indexes = self.get_sensor_data_indexes(step)
            state[f'{sensor}_indexes'] = indexes
            if retrieve_data:
                data = self.get_sensor_data(sensor, step)
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

    def get_sensor_data(self, sensor, step):
        if len(self.sensors[sensor]['data']) < self.no_data_points:
            data = [self.sensors[sensor]['data'][0] for i in
                    range(self.no_data_points - len(self.sensors[sensor]['data']))] \
                   + [frame for frame in self.sensors[sensor]['data']]
        else:
            data = self.sensors[sensor]['data'][-self.no_data_points:]
        return data

    def get_sensor_data_indexes(self, step):
        if step < self.no_data_points:
            indexes = [0 for i in range(self.no_data_points - step)] \
                   + [idx + 1 for idx in range(step)]
        else:
            indexes = [idx for idx in range(step-self.no_data_points, step)]
        return indexes


    def initialize_vehicle(self):
        if not self.initialized:
            self.actor:carla.Vehicle = self.world.spawn_actor(self.actor, numpy_to_transform(self.spawn_point))
            self.actor.apply_control(carla.VehicleControl(brake=1., gear=1))
            self.initialized = True
            print('Vehicle initilized')
        else:
            raise Exception('Vehicle already spawned')

    def initialize_sensors(self):

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

    def _release_control(self):
        self.actor.apply_control(carla.VehicleControl(throttle=0., brake=0., gear=1))

    def _release_data(self, step: int) -> None:
        '''
        Function which saves sensor data to the disk and releases memory.

        :param step:
        :return: None
        '''
        for sensor in self.sensors.keys():
            if (sensor is not 'collisions'):
                save_path = f'{DATA_PATH}/experiments/{self.map}/{DATE_TIME}_{self.controller.__class__.__name__}/{sensor}_{step}.png'
                save_img(img=self.sensors[sensor]['data'][-1], path=save_path)
                if step > 4:
                    self.sensors[sensor]['data'].pop(0)

    def destroy(self, data:bool=False):
        '''
        Destroying agent entities while preserving sensors data.
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
    '''
    To orchestrate asynchronous agents and world ticks in future
    '''
    def __init__(self, client:carla.Client):
        self.client = client
        # self.world = None
        # self.agents = []


    def reset_env(self, args:argparse.ArgumentParser):
        # [agent.actor.destroy() for agent in self.agents]
        # self.agents = []
        world: carla.World = self.client.load_world(args.map)
        if args.synchronous:
            settings = world.get_settings()
            settings.synchronous_mode = True  # Enables synchronous mode
            settings.fixed_delta_seconds = 1 / args.frames
            world.apply_settings(settings)
        return world

    def toggle_world(self, world:carla.World, frames:int=FRAMERATE):
        settings = world.get_settings()
        settings.synchronous_mode = not settings.synchronous_mode
        settings.fixed_delta_seconds = abs(float(settings.fixed_delta_seconds or 0) - 1/frames)
        world.apply_settings(settings)

    def calc_reward(self, points_3D:np.array, state:dict, next_state, alpha: float = .99, step: int = 0):

        if calc_distance(actor_location=next_state['location'], points_3D=points_3D) > calc_distance(
                actor_location=state['location'], points_3D=points_3D):
            return - (next_state['velocity'] / (state['velocity'] + 1)) * (alpha ** step)

        # return (next_state['velocity']/(state['velocity'] + 1)) * \
        #        ((tracklen - calc_from_closest_distance) - distance_travelled )  * (alpha**step)
        return (next_state['velocity'] / (state['velocity'] + 1)) * (alpha ** step)