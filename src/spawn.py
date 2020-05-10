import carla
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev

from carla import Transform, Location, Rotation

numpy_to_transform = lambda point: Transform(Location(point[0], point[1], point[2]), Rotation(yaw=point[3], pitch=0, roll=0))
transform_to_numpy = lambda transform: np.array([transform.location.x, transform.location.y, transform.location.z, transform.rotation.yaw])
numpy_to_location = lambda point: Location(point[0], point[1], point[2])
location_to_numpy = lambda location: np.array([location.x, location.y, location.z])
velocity_to_kmh = lambda v: float(3.6 * np.math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

# porównaj prędkość z funkcji, z wyliczonej z przemieszczenia między klatkami

def df_to_spawn_points(data: pd.DataFrame, n:int=10000, inverse:bool=False) -> list:

    pts_3D = data[['x', 'y', 'z']].values.T
    if inverse:
        pts_3D = pts_3D.flipud(axis=0)
    tck, u = splprep(pts_3D, u=None, s=1.5, per=1, k=2)
    u_new = np.linspace(u.min(), u.max(), n+1)
    x, y, z = splev(u_new, tck, der=0)
    pts_3D = np.c_[x,y,z]
    yaws = [calc_azimuth(pointA, pointB) for pointA, pointB in zip(pts_3D[:-1], pts_3D[1:])]

    return np.c_[pts_3D[:-1], yaws]
    # sp_points = iter(np.c_[pts_3D[:-1], yaws])
    # return [Transform(Location(x, y, z), Rotation(yaw=yaw, pitch=0, roll=0)) for (x, y, z, yaw) in sp_points]



def calc_azimuth(pointA:tuple, pointB:tuple) -> float:
    sin_alpha = pointB[1] - pointA[1]
    cos_alpha = pointB[0] - pointA[0]

    alpha = np.degrees(np.arctan2(sin_alpha, cos_alpha))

    return alpha

def calc_speed(actor:carla.Vehicle, framerate:int) -> float:
    vehicle_location = location_to_numpy(actor.get_location())
    speed = location_to_numpy(actor.get_velocity()) # -> jako prędkość na sekundę? odległość przebyta w ciągu następnej sekundy? czy poprzedniej?

    #Współrzędne metryczne, różnica między klatkami = ilość pokonanych metrów w klatce -> * klatki na sekundę * 1000/3600 -> km/h

    v = actor.get_velocity()
    kmh = float(3.6 * np.math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

    distance = np.linalg.norm([vehicle_location-speed]) #meters/frame
    speed_kmh = distance * framerate * 1000/3600

def to_vehicle_control(gas_brake:float, steer:float) -> carla.VehicleControl:
    if gas_brake > 0:
        return carla.VehicleControl(throttle = gas_brake, steer=steer, reverse=False)
    elif (gas_brake < 0.5) & (gas_brake > 0.2) :
        return carla.VehicleControl(throttle=0, steer=steer, reverse=False)
    elif (gas_brake < 0.2) & (gas_brake > -0.5) :
        return carla.VehicleControl(throttle=0, brake=-gas_brake-0.5, steer=steer, reverse=False)
    else:
        return carla.VehicleControl(throttle = -gas_brake, steer=-steer, reverse=True)

def set_spectator_above_actor(world, actor):
    spectator = world.get_spectator()
    trasform = carla.Transform((actor.get_location() + carla.Location(0, 0, 15)),
                    carla.Rotation(pitch=-15, yaw=actor.get_transform().rotation.yaw))
    spectator.set_transform(trasform)

