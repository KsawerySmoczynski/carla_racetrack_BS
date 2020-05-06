import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev

from carla import Transform, Location, Rotation


def df_to_spawn_points(data: pd.DataFrame, n:int=10000, inverse:bool=False) -> list:

    pts_3D = data[['x', 'y', 'z']].values.T
    if inverse:
        pts_3D = pts_3D.flip(axis=0)
    tck, u = splprep(pts_3D, u=None, s=1.5, per=1, k=2)
    u_new = np.linspace(u.min(), u.max(), n+1)
    x, y, z = splev(u_new, tck, der=0)
    pts_3D = np.c_[x,y,z]
    yaws = [calc_azimuth(pointA, pointB) for pointA, pointB in zip(pts_3D[:-1], pts_3D[1:])]
    sp_points = iter(np.c_[pts_3D[:-1], yaws])

    return [Transform(Location(x, y, z), Rotation(yaw=yaw, pitch=0, roll=0)) for (x, y, z, yaw) in sp_points]

def calc_azimuth(pointA:tuple, pointB:tuple) -> float:
    sin_alpha = pointB[1] - pointA[1]
    cos_alpha = pointB[0] - pointA[0]

    alpha = np.degrees(np.arctan2(sin_alpha, cos_alpha))

    return alpha