import time
import argparse

import numpy as np
import pandas as pd

import carla

#Local imports
from spawn import df_to_spawn_points, numpy_to_transform
from control.mpc_control import MPCController


#Configs
from config import CARLA_IP, DATA_PATH, STORE_DATA
from config import toggle_world


def run_client(args):
    # create client -> config client does that
    #   check if loaded map is target map, if true proceed if false try to load desired world if other actor on map raise error -> or write error and close

    # Connecting to client -> later package it in function which checks if the world is already loaded and if the settings are the same.
    # In order to host more scripts concurrently
    # client = carla.Client(args.host, args.port)
    client = carla.Client('localhost', 2000) #local
    client.set_timeout(5.0)  # seconds
    # load world desired condition -> config client does that
    world = client.load_world(args.map)
    if args.synchronous:
        settings = world.get_settings()
        settings.synchronous_mode = True# Enables synchronous mode
        settings.fixed_delta_seconds = 1/args.frames
        world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('*aud*')[0] # -> change it to be parametric, maybe shuffle each time to add robustness

    # create config dict for raport
    #
    # Here let's create data structure which will let us save summary results from each run_episode iteration
    # for ex. status, distance travelled, reward obtained -> may be dataframe, we'll append each row after iteration


    # load spawnpoints from csv -> generate spawn points from notebooks/20200414_setting_points.ipynb
    spawn_points_df = pd.read_csv(f'{DATA_PATH}/spawn_points/{args.map}.csv')
    spawn_points = df_to_spawn_points(spawn_points_df, n=1000, inverse=False) #We keep it here in order to have one way simulation within one script

    #Sensors initialization
    sensors = {}
    depth_bp = blueprint_library.find('sensor.camera.depth')
    relative_transform = carla.Transform(carla.Location(1.0, 0, 1.4), carla.Rotation(-5., 0, 0))
    cc = carla.ColorConverter.LogarithmicDepth
    sensors['depth'] = {'blueprint': depth_bp,
                        'transform': relative_transform,
                        'color_converter':cc}

    # Controller initialization
    if args.controller is 'MPC':
        controller = MPCController(target_speed=60.)

    # episodes loop
    for episode_idx in range(args.num_episodes): # may be useless at this moment
        # Attach sensor to car -> has to save to global structure, maybe all of this should be inside run episode

        status, actor_dict, env_dict, sensor_data = run_episode(world, controller, vehicle_bp, sensors, spawn_points, args)

    # Save imgs, dicts, waypoints
    #
    #   run_episode -> status, imgs, states-action dicts, distance
    #
    #       #saving state-action dicts + imgs in order to feed DQN in off-policy mode.
    # save info to report
    print('Finito la bamba')
    pass


def run_episode(world:carla.World, controller, vehicle_bp:carla.ActorBlueprint,
                sensors:dict, spawn_points:np.array, args) -> (str, dict, dict, list):
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
    NUM_STEPS = args.num_steps
    depth_data = []
    collisions_data = []

    # Ordering spawn points
    start_point_idx = int(np.random.randint(len(spawn_points))) # -> save for logging
    waypoints = np.concatenate([spawn_points[start_point_idx:, :], spawn_points[:start_point_idx, :]]) #delete yaw column

    # SPAWN ACTOR -> TIC AS MANY TIMES FOR HIM TO STABILIZE
    actor = world.spawn_actor(vehicle_bp, numpy_to_transform(spawn_points[start_point_idx]))
    actor.apply_control(carla.VehicleControl(brake=1.))
    spectator = world.get_spectator()
    spectator.set_transform(numpy_to_transform(spawn_points[start_point_idx - 5]))
    actor_pos = None
    world.tick()
    world.tick()
    # Calculate norm of all cordinates
    n_tics = 0
    while actor_pos != actor.get_location():
            # time.sleep(1.)
        actor_pos = actor.get_location()
        world.tick()
        n_tics += 1
    # not really
    actor.destroy()
    world.tick()

    #ATTACH SENSORS
    to_array = lambda img: np.asarray(img.raw_data, dtype=np.int16).reshape(img.height, img.width, -1)
    depth_camera = world.spawn_actor(sensors['depth']['blueprint'], sensors['depth']['transform'], attach_to=actor) #rigid attachment by default
    depth_camera.listen(lambda img_raw: (img_raw.convert(sensors['depth']['color_converter']), depth_data.append(to_array(img_raw)))) # o lol działa
    # sensors['depth']['actor'] = depth_camera

    # Release handbrake
    world.tick()  # x4? allow controll each 4 frames
    for step in range(NUM_STEPS):
        # Pack everything into Agent class



        #Retrive sensorical data, retrive location data
        #if statements regarding colisions and if actor get stuck
        #if it is the case return fail message

        #Were taking 4 last frames
        if len(depth_data) < 4:
            frames = [depth_data[0] for i in range(4-len(depth_data))] + [frame for frame in depth_data]
        else:
            frames = depth_data[step-4:step]

        # actuators_info = actor.ge
        sensors_info = {'depth': frames,
                        'collisions': None}


        # remember that distance to the finish should be known to NN actor, as a part of actuator, or it shouldn't?
        steer, gas_brake = controller.control(
            actor,
            waypoints,
            sensors_info
        )

        # Actor apply control
        # Tick
        # Calculate reward based on distance
        # append s, a, r, s' to logging structures -> states can be indexes in sensor data storages
        # calculate for ex. distance and add to separate informative logging structure

    if STORE_DATA:
        sensor_data = {'depth': depth_data,
                       'collisions': collisions_data}
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
        default=30,
        type=float,
        help='Number of frames per second, dont set below 10, use with --synchronous flag only')

    #World configs
    argparser.add_argument(
        '--map',
        metavar='M',
        default='circut_spa',
        help='Avialable maps: "circut_spa", "RaceTrack", "Racetrack2". Default: "circut_spa"')

    # Simulation
    argparser.add_argument(
        '-e', '--num_episodes',
        default=1,
        type=int,
        dest='num_episodes',
        help='Number of episodes')

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