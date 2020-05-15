import json
import numpy as np
import carla
import visdom
from tensorboardX import SummaryWriter

from config import IMAGE_DOWNSIZE_FACTOR, DATE_TIME
from spawn import location_to_numpy, calc_azimuth, velocity_to_kmh

to_array = lambda img: np.asarray(img.raw_data, dtype=np.int16).reshape(img.height, img.width, 4)  # 4 because image is in BRGB format
to_rgb_resized = lambda img: img[..., :3][::IMAGE_DOWNSIZE_FACTOR, ::IMAGE_DOWNSIZE_FACTOR, ::-1]  # making it RGB from BRGB with [...,:3][...,::-1]



def closest_checkpoint(actor:carla.Vehicle, checkpoints:np.array):
    actor_location = location_to_numpy(actor.get_location())
    distances = np.linalg.norm(checkpoints-actor_location, axis=1)
    azimuths = [abs(calc_azimuth(actor_location[:2], point[:2])) for point in checkpoints]

    # TODO be clever -> this is shit.
    cut = np.argmin(distances)
    if np.argmin(azimuths) >= cut+1:
        return checkpoints[cut+1]
    else:
        return checkpoints[cut]



def calc_distance(actor_location:np.array, points_3D:np.array, cut:float=0.02) -> float:
    '''
    Returns distance along #D points to the last one, skipping n closest points
    :param actor: np.array
    :param points_3D: np.array, shape = (n-points, ndimensions)
    :param cut: int, how big percentage of whole racetrack length is being skipped in calculating distance to finish,
                        values higher than 0.025 aren't recommended
    :return: float - distance to the last point.
    '''

    skipped_points = np.argmin(np.linalg.norm(points_3D-actor_location, axis=1))
    cut_idx = int(points_3D.shape[0] * cut)
    skip = (skipped_points + cut_idx) % points_3D.shape[0]
    actor_to_point = np.linalg.norm(points_3D[skip] - actor_location)
    points_deltas = np.diff(points_3D[skip:], axis=0)
    distance = actor_to_point + np.sqrt((points_deltas**2).sum(axis=1)).sum()

    return distance

def visdom_initialize_windows(viz:visdom.Visdom, sensors:dict, location):

    windows = {}
    if sensors['depth']:
        windows['depth'] = viz.image(np.zeros((3, 75, 100)), opts=dict(caption='Depth', title='Depth sensor'))

    if sensors['rgb']:
        windows['rgb'] = viz.image(np.zeros((3, 75, 100)), opts=dict(caption='RGB', title='RGB camera'))

    windows['trace'] = viz.line(X=[location[0]], Y=[location[1]], opts=dict(caption='Trace', title='Actor trace'))
    windows['reward'] = viz.line(X=[0], Y=[0], opts=dict(caption='Rewards', title='Rewards received'))
    windows['velocity'] = viz.line(X=[0], Y=[0], opts=dict(caption='Velocity', title='Velocity in kmh'))
    windows['gas_brake'] = viz.line(X=[0], Y=[0], opts=dict(caption='Gas and brake', title='Gas and brake'))
    windows['steer'] = viz.line(X=[0], Y=[0], opts=dict(caption='Steer', title='Steer angle'))

    return windows

def visdom_log(viz:visdom.Visdom, windows:dict, sensors:dict, state:dict, action:dict, reward:float, step:int) -> None:
    '''

    :param viz:
    :param windows:
    :param sensors:
    :param state:
    :param action:
    :param reward:
    :param step:
    :return:
    '''
    viz.line(X=[state['location'][0]], Y=[state['location'][1]], win=windows['trace'], update='append')
    viz.line(X=[step], Y=[reward], win=windows['reward'], update='append')
    viz.line(X=[step], Y=[action['gas_brake']], win=windows['gas_brake'], update='append')
    viz.line(X=[step], Y=[action['steer']], win=windows['steer'], update='append')
    viz.line(X=[step], Y=[state['velocity']], win=windows['velocity'], update='append')

    if 'depth' in sensors.keys():
        img = sensors['depth'][-1]
        img = np.moveaxis(img, 2, 0).copy().astype(np.uint8)
        viz.image(img=img, win=windows['depth'])

    if 'rgb' in sensors.keys():
        img = sensors['rgb'][-1]
        img = np.moveaxis(img, 2, 0).copy().astype(np.uint8)
        viz.image(img=img, win=windows['rgb'])


def tensorboard_log(title:str, writer:SummaryWriter, state:dict, action:dict, reward:float, step:int) -> None:
    '''
    Write logging info to tensorboard writer
    :param writer:
    :param state:
    :param action:
    :param reward:
    :param step:
    :return: None
    '''

    writer.add_scalar(tag=f'{title}/distance_2finish', scalar_value=state['distance_2finish'], global_step=step)
    writer.add_scalar(tag=f'{title}/gas_brake', scalar_value=action['gas_brake'], global_step=step)
    writer.add_scalar(tag=f'{title}/steer', scalar_value=action['steer'], global_step=step)
    writer.add_scalar(tag=f'{title}/velocity', scalar_value=state['velocity'],
                      global_step=step)
    writer.add_scalar(tag=f'{DATE_TIME}/reward', scalar_value=reward, global_step=step)

def save_episode_info(status:str, actor_dict:dict, env_dict:dict, sensor_data:dict):
    pass


#TODO liczymy przebytą odległość i odległość po punktach, różnica jest nagrodą
# def calc_distance(actor_loc:np.array, points_3D:np.array, checkpoint:np.array) -> float:
#     #TODO pass closest checkpoint index instead of passing points choose closest one based on azimuth.
#     # Take 2 closest checkpoints and take one with lower azimuth
#     # Dynamically calculate distance according to checkpoints so if the car turns around it will receive negative reward
#     # As it's not diminishing the distance to the next point
#
#     '''
#     Returns distance along #D points to the last one, skipping n closest points
#     :param actor_loc: carla.Vehicle
#     :param points_3D: np.array, shape = (n-points, ndimensions)
#     :param cut: int, how big percentage of whole racetrack length is being skipped in calculating distance to finish,
#                         values higher than 0.025 aren't recommended
#     :return: float - distance to the last point.
#     '''
#
#     skipped_points = np.where(points_3D == checkpoint)[0][0] # -> dont hardcode
#     actor_to_point = np.linalg.norm(points_3D[skipped_points] - actor_loc)
#     points_deltas = np.diff(points_3D[skipped_points:], axis=0)
#     distance = actor_to_point + np.sqrt((points_deltas**2).sum(axis=1)).sum()
#
#     return distance

# def generate_checkpoints(points_3D:np.array, cut:float=0.02):
#     '''
#     Selecting checkpoints along the racetrack
#     :param points_3D:
#     :param cut: float,
#     :return:
#     '''
# #     USE NP where
#     cut_idx = int(points_3D.shape[0] * cut)
#     # We're cutting first points with len of cut_index in order to have first checkpoint =! start point
#     checkpoints = points_3D[::cut_idx,:]
#
#     return checkpoints