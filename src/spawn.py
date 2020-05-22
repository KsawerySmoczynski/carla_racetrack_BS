import carla
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev

from carla import Transform, Location, Rotation

#Easy selfexplaining lambdas
numpy_to_transform = lambda point: Transform(Location(point[0], point[1], point[2]), Rotation(yaw=point[3], pitch=0, roll=0))
transform_to_numpy = lambda transform: np.array([transform.location.x, transform.location.y, transform.location.z, transform.rotation.yaw])
numpy_to_location = lambda point: Location(point[0], point[1], point[2])
location_to_numpy = lambda location: np.array([location.x, location.y, location.z])
velocity_to_kmh = lambda v: float(3.6 * np.math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
numpy_to_velocity_vec = lambda v: carla.Vector3D(x=v[0], y=v[1], z=v[2])


def df_to_spawn_points(data: pd.DataFrame, n:int=10000, inverse:bool=False) -> np.array:
    '''
    Method converting spawnpoints loaded from DataFrame into equally placed points on the map.
    :param data:pd.Dataframe, handcrafted points in tabular form
    :param n:int number of consecutive generated points
    :param inverse: if to inverse direction of the racetrack
    :return:np.array
    '''
    pts_3D = data[['x', 'y', 'z']].values
    if inverse:
        pts_3D = np.flipud(pts_3D)
    pts_3D = pts_3D.T
    tck, u = splprep(pts_3D, u=None, s=1.5, per=1, k=2)
    u_new = np.linspace(u.min(), u.max(), n+1)
    x, y, z = splev(u_new, tck, der=0)
    pts_3D = np.c_[x,y,z]
    yaws = [calc_azimuth(pointA, pointB) for pointA, pointB in zip(pts_3D[:-1], pts_3D[1:])]

    return np.c_[pts_3D[:-1], yaws]


def calc_azimuth(pointA:tuple, pointB:tuple) -> float:
    '''
    Calculating azimuth betweed two points, azimuth returned in degrees in range <-180, 180>
    :param pointA:tuple in form (x, y) float coordinates
    :param pointB:tuple in form (x, y) float coordinates
    :return:float
    '''
    sin_alpha = pointB[1] - pointA[1]
    cos_alpha = pointB[0] - pointA[0]

    alpha = np.degrees(np.arctan2(sin_alpha, cos_alpha))

    return alpha

def to_vehicle_control(gas_brake:float, steer:float) -> carla.VehicleControl:
    #TODO think about it
    '''
    Modelling inputs from controller to actuator values.
    :param gas_brake:float in range <-1,1>
    :param steer:float in range <-1,1>
    :return: carla.VehicleControl
    '''
    #TODO -> podziel na 4 obszary i przeskaluj dynamikę
    if gas_brake > 0.5:
        return carla.VehicleControl(throttle = 2*gas_brake-1, steer=steer, reverse=False)
    elif (gas_brake < 0.5) & (gas_brake > 0.) :
        return carla.VehicleControl(throttle=0, steer=steer, reverse=False)
    elif (gas_brake < 0.) & (gas_brake > -0.5) :
        return carla.VehicleControl(throttle=0, brake=-2*gas_brake-1, steer=steer, reverse=False)
    else:
        return carla.VehicleControl(throttle=-2*gas_brake-1, steer=-steer, reverse=True)

def set_spectator_above_actor(spectator:carla.Actor, transform:np.array) -> None:
    '''
    Changes position of the spectator relative to actor position.
    :param spectator:
    :param transform:
    :return:
    '''
    transform = numpy_to_transform(transform + [0, 0, 15, 0])
    transform.rotation.pitch = -15
    spectator.set_transform(transform)


def sensors_config(blueprint_library:carla.BlueprintLibrary,
                   depth:bool=True, rgb:bool=False, collision:bool=True,) -> dict:
    '''
    Configures sensors blueprints, relative localization and transformations related to sensor.
    :param blueprint_library:carla.BlueprintLibrary
    :param depth:bool
    :param collision:bool
    :param rgb:bool
    :return: sensors:dict
    '''
    sensors = {}
    if depth:
        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_relative_transform = carla.Transform(carla.Location(1.4, 0, 1.4), carla.Rotation(-5., 0, 0))
        cc = carla.ColorConverter.LogarithmicDepth
        sensors['depth'] = {'blueprint': depth_bp,
                            'transform': depth_relative_transform,
                            'color_converter':cc}

    if rgb:
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_relative_transform = carla.Transform(carla.Location(1.4, 0, 1.4), carla.Rotation(-5., 0, 0))
        sensors['rgb'] = {
            'blueprint': rgb_bp,
            'transform': rgb_relative_transform,
                                    }

    if collision:
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_relative_transform = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(0, 0, 0))
        sensors['collisions'] = {
            'blueprint': collision_bp,
            'transform': collision_relative_transform
                                }

    return sensors
