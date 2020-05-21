import json
import os
import time
from statistics import mean
import numpy as np
import pandas as pd
import visdom as vis
from PIL import Image
from ast import literal_eval

from config import DATE_TIME, SENSORS, EXPERIMENTS_PATH, MAP, INVERSE


def vis_initialize_windows(visdom:vis.Visdom, sensors:dict):

    windows = {sensor: visdom.image(np.zeros((3, 75, 100)), opts=dict(title=f'{DATE_TIME} {sensor} sensor', width=800, height=600)) \
               for sensor, value in sensors.items() if value & (sensor is not 'collisions')}

    windows['trace'] = visdom.line(X=[0], Y=[0], opts=dict(title=f'{DATE_TIME} Actor trace'))
    windows['reward'] = visdom.line(X=[0], Y=[0], opts=dict(title=f'{DATE_TIME} Rewards received'))
    windows['velocity'] = visdom.line(X=[0], Y=[0], opts=dict(title=f'{DATE_TIME} Velocity in kmh'))
    windows['gas_brake'] = visdom.line(X=[0], Y=[0], opts=dict(title=f'{DATE_TIME} Gas and brake'))
    windows['steer'] = visdom.line(X=[0], Y=[0], opts=dict(title=f'{DATE_TIME} Steer angle'))
    windows['distance_2finish'] = visdom.line(X=[0], Y=[0], opts=dict(title=f'{DATE_TIME} Distance 2finish'))

    return windows


def vis_log_data(visdom:vis.Visdom, windows:dict, data_path:str) -> int:
    '''

    :param viz:
    :param windows:
    :param data_path:
    :return:
    '''
    #Dont load whole file only last row, and append last row
    data = pd.read_csv(f'{data_path}/episode_info.csv')
    data['location'] = [literal_eval(item) for item in data['location']]
    data['velocity_vec'] = [literal_eval(item) for item in data['velocity_vec']]

    location_x = [x[0] for x in data['location']]
    location_y = [x[1] for x in data['location']]

    visdom.line(X=location_x, Y=location_y, win=windows['trace'], update='replace',
                opts={'ytickmin': min(location_y)-0.1*mean(location_y),
                    'ytickmax': max(location_y)+0.1*mean(location_y),
                    'xtickmin': min(location_x)-0.1*mean(location_x),
                    'xtickmax': max(location_x)+0.1*mean(location_x)})

    for value in ['reward', 'gas_brake', 'steer', 'velocity', 'distance_2finish']:
        opts = {'ytickmin': min(data[value]) - 0.1 * data[value].mean(),
                'ytickmax': max(data[value]) + 0.1 * data[value].mean(),
                'xtickmin': min(data['step']) - 0.05 * data['step'].mean(),
                'xtickmax': max(data['step']) + 0.05 * data['step'].mean()}
        visdom.line(X=data['step'], Y=data[value], win=windows[value], update='replace',
                    opts= opts)

    return list(data['step'])[-1]


def vis_log_photos(visdom:vis, windows:dict, data_path:str, sensors:dict):
    '''
    Show most recent frame
    :param visdom:
    :param windows:
    :param data_path:
    :param sensors:
    :return:
    '''
    idx = max([int(x.split('_')[-1][:-4]) for x in os.listdir(f'{data_path}/sensors')])
    for sensor in [sensor for sensor, value in sensors.items() if value & (sensor is not 'collisions')]:
        img = np.array(Image.open(f'{data_path}/sensors/{sensor}_{idx}.png')).astype(np.uint8)
        img = np.moveaxis(img, 2, 0)
        visdom.image(img=img, win=windows[sensor], opts=dict(title=json.loads(visdom.get_window_data(windows[sensor]))['title'], width=800, height=600))


def vis_run(visdom:vis.Visdom, data_path:str, photos_refresh:float=0.2, data_rate:float=5):
    windows = vis_initialize_windows(visdom, SENSORS)
    passed_time = 0
    step = 0
    while True:
        vis_log_photos(visdom=visdom, windows=windows, data_path=data_path, sensors=SENSORS)
        if passed_time >= data_rate:
            data_step = vis_log_data(visdom=visdom, windows=windows, data_path=data_path)
            if step == data_step:
                break
            step = data_step
            passed_time = 0
        passed_time += photos_refresh
        time.sleep(photos_refresh)


def main():


    visdom = vis.Visdom(port=6006)

    experiment = sorted(os.listdir(f'{EXPERIMENTS_PATH}/{MAP}{"_inverse"*INVERSE}'))[-1]
    controller = sorted(os.listdir(f'{EXPERIMENTS_PATH}/{MAP}{"_inverse"*INVERSE}/{experiment}'))[-1]
    vis_run(visdom=visdom, data_path=f'{EXPERIMENTS_PATH}/{MAP}{"_inverse"*INVERSE}/{experiment}/{controller}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')