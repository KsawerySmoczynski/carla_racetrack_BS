import os

from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import carla
import visdom
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from config import IMAGE_DOWNSIZE_FACTOR, DATE_TIME, IMAGE_SIZE, EXTRA_REWARD, GAMMA
from spawn import location_to_numpy, calc_azimuth

to_rgb_pil = lambda img: Image.frombuffer(mode='RGBA', size=IMAGE_SIZE, data=img.raw_data.tobytes()).convert('RGB')
to_array = lambda img: np.asarray(img.raw_data, dtype=np.int8).reshape(img.height, img.width, 4)  # 4 because image is in BRGB format
to_rgb = lambda img: img[..., :3][..., ::-1]  # making it RGB from BRGB with [...,:3][...,::-1]
to_resize = lambda img: img[..., :3][::IMAGE_DOWNSIZE_FACTOR, ::IMAGE_DOWNSIZE_FACTOR, ::-1]  # making it RGB from BRGB with [...,:3][...,::-1]


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
    :return: float - distance to the last point as a fraction of whole track
    '''

    skipped_points = np.argmin(np.linalg.norm(points_3D-actor_location, axis=1))
    cut_idx = int(points_3D.shape[0] * cut)
    if (skipped_points + cut_idx) < points_3D.shape[0]:
        skip = (skipped_points + cut_idx) % points_3D.shape[0]
    else:
        skip = skipped_points
    actor_to_point = np.linalg.norm(points_3D[skip] - actor_location)
    points_deltas = np.diff(points_3D[skip:], axis=0)
    distance = actor_to_point + np.sqrt((points_deltas**2).sum(axis=1)).sum()

    return round(distance, 5)

def calc_track_azimuth(actor_location:np.array, points_3D:np.array, cut:float=0.01) -> float:
    skipped_point = np.argmin(np.linalg.norm(points_3D - actor_location, axis=1))
    cut_idx = int(points_3D.shape[0] * cut)

    if (skipped_point + cut_idx) < points_3D.shape[0]:
        skip = (skipped_point + cut_idx) % points_3D.shape[0]
    else:
        skip = skipped_point
    track_point = points_3D[skip][:2]

    return calc_azimuth(actor_location[:2], track_point)



def visdom_initialize_windows(viz:visdom.Visdom, title:str, sensors:dict, location):
    '''
    Deprecated
    :param viz:
    :param title:
    :param sensors:
    :param location:
    :return:
    '''
    windows = {}
    if sensors['depth']:
        windows['depth'] = viz.image(np.zeros((3, 75, 100)), opts=dict(title=f'{title} Depth sensor', width=800, height=600))

    if sensors['rgb']:
        windows['rgb'] = viz.image(np.zeros((3, 75, 100)), opts=dict(title=f'{title} RGB camera', width=800, height=600))

    windows['trace'] = viz.line(X=[location[0]], Y=[location[1]], opts=dict(title=f'{title} Actor trace'))
    windows['reward'] = viz.line(X=[0], Y=[0], opts=dict(title=f'{title} Rewards received'))
    windows['velocity'] = viz.line(X=[0], Y=[0], opts=dict(title=f'{title} Velocity in kmh'))
    windows['gas_brake'] = viz.line(X=[0], Y=[0], opts=dict(title=f'{title} Gas and brake'))
    windows['steer'] = viz.line(X=[0], Y=[0], opts=dict(title=f'{title} Steer angle'))
    windows['distance_2finish'] = viz.line(X=[0], Y=[0], opts=dict(title=f'{title} Distance 2finish'))

    return windows


def visdom_log(viz:visdom.Visdom, windows:dict, state:dict, action:dict, reward:float, step:int) -> None:
    '''
    Deprecated
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
    viz.line(X=[step], Y=[state['distance_2finish']], win=windows['distance_2finish'], update='append')

    if 'depth' in state.keys():
        img = state['depth'][-1]
        img = np.moveaxis(img, 2, 0).copy().astype(np.uint8)
        viz.image(img=img, win=windows['img'], opts=dict(width=800, height=600))

    if 'rgb' in state.keys():
        img = state['rgb'][-1]
        img = np.moveaxis(img, 2, 0).copy().astype(np.uint8)
        viz.image(img=img, win=windows['rgb'], opts=dict(width=800, height=600))


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


def save_img(img:np.array, path:str, mode:str='RGB') -> None:
    '''
    Simple function for saving pictures in np.array format (HEIGHT, WIDTH, CHANNELS)

    :param img: np.array, (H, W, C) format
    :param path: path to saved file
    :param mode: 'L' for one channel grayscale, 'LA' for 2 channel grayscale, default: 'RGB'
    :return:
    '''
    if len(path.split('/')) > 1:
        os.makedirs(name='/'.join(path.split('/')[:-1]), exist_ok=True)
    Image.fromarray(obj=img, mode=mode).save(fp=path)


def init_reporting(path:str, sensors:dict) -> None:
    '''
    Initialize file for logging based on suite of utilized sensors
    :param path:str, path to the experiment folder
    :param sensors: dict of sensors from config consisting boolean values
    :return: None
    '''

    if len(path.split('/')) > 1:
        os.makedirs(name='/'.join(path.split('/')), exist_ok=True)

    header = 'step'
    sensors_header = ','.join([f'{k * v}_indexes' for k,v in sensors.items() if k is not 'collisions'])
    if 'collisions' in sensors.keys():
        sensors_header = f'{sensors_header},collisions'
    header = f'{header},{sensors_header}'
    # header = f'{header},velocity,velocity_vec,yaw,location,distance_2finish,steer,gas_brake,reward\n'
    header = f'{header},state_steer,state_gas_brake,velocity,velocity_vec,yaw,location,distance_2finish,steer,gas_brake,reward\n'

    with open(f'{path}/episode_info.csv', 'w+') as file:
        file.write(header)
    print('Init succesfull')


def save_info(path:str, state:dict, action:dict, reward:float, done:int=0) -> pd.DataFrame:
    '''
    Appends information after every step about state, actions and received reward
    :param path: str, path to experiment folder
    :param state: dict, state dictionary
    :param action:dict, action dictionary
    :param reward: float, reward value
    :return: None
    '''
    info = {**state, **action, 'reward':reward, 'done':done}
    # info = {**state, **action, 'reward':reward}
    info = pd.DataFrame().from_dict({k:[v] for k,v in info.items() if 'data' not in k})
    info_csv = info.to_csv(index=False, header=False)
    with open(f'{path}/episode_info.csv', 'a') as file:
        file.write(info_csv)

    return info


def save_terminal_state(path:str, state:dict, action:dict):
    action['steer'] = 0.
    action['gas_brake'] = 0.
    if 'q_pred' in list(action.keys()):
        action['q_pred'] = 0.
    save_info(path=path, state=state, action=action, reward=0, done=1)


def update_Qvals(path:str) -> None:
    '''
    Adds qvalues to the saved dataframes
    :param path:
    :return:
    '''
    df = pd.read_csv(f'{path}/episode_info.csv')
    df['q'] = [sum(df['reward'][i:]) for i in range(df.shape[0])]
    df.to_csv(f'{path}/episode_info.csv', index=False)


def plot_gray(img):
    '''
    img has to have shape (1, H, W)
    :param img:
    :return:
    '''
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


def arg_bool(arg:str):
    arg = arg.lower()
    if arg in ['true', 't', '1', 'y']:
        return True
    else:
        return False


def clean_mpc_bias(path, save:bool=False):
    df = pd.read_csv(f'{path}/episode_info.csv')
    df[:24]['steer'] = [0. for i in range(24)]
    if save:
        df.to_csv(f'{path}/episode_info.csv', index=False)
    else:
        return df


def add_dones_and_terminal_state(df:pd.DataFrame, path:str=None):
    if 'dones' in df.columns:
        return None
    if df.loc[len(df) - 1, 'collisions'] != 0:
        df.loc[len(df) - 2, 'reward'] = df.loc[len(df) - 1, 'reward']
    if df.loc[len(df) - 1, 'distance_2finish'] <= 50:
        df.loc[len(df) - 2, 'reward'] = EXTRA_REWARD*GAMMA**df.loc[len(df) - 2, 'step']
    df.loc[len(df) - 1, 'reward'] = .0
    df.loc[len(df) - 1, 'steer'] = .0
    df.loc[len(df) - 1, 'gas_brake'] = .0
    df.loc[len(df) - 1, 'q'] = .0
    dones = [0 for i in range(len(df)-1)] + [1]
    df['done'] = dones
    df['q'] = [sum(df['reward'][i:]) for i in range(df.shape[0])]
    if path:
        df.to_csv(f'{path}/episode_info.csv', index=False)
    else:
        return df

def change_reward_scheme(df:pd.DataFrame, gamma: float = GAMMA, punishment:float=0.01) -> pd.DataFrame:

    def reward(dist, next_dist):
        if dist > next_dist:
            return 1
        elif dist == next_dist:
            return 0
        else:
            return -1
    rewards = [reward(df.loc[i, 'distance_2finish'], df.loc[i+1, 'distance_2finish'])*gamma**i - punishment
               for i in range(len(df)-1)] + [0]
    df['reward'] = rewards
    qs = [sum(df.loc[i:, 'reward']) for i in range(len(df))]
    df['q'] = qs
    return df

