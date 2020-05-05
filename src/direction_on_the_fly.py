import os
import sys
sys.path.append(['..'])

import numpy as np
import math
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.express as px
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

DATA_PATH = '/home/ksawi/Documents/Workspace/carla/carla_racetrack_BA/data/spawn_points'
TRACKS = ['circut_spa', 'RaceTrack', 'RaceTrack2']

track_DF = pd.read_csv(f'{DATA_PATH}/{TRACKS[0]}.csv')

x = track_DF['x'].values
y = track_DF['y'].values
z = track_DF['z'].values
okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
xp = np.r_[x[okay], x[-1], x[0]]
yp = np.r_[y[okay], y[-1], y[0]]

jump = np.sqrt(np.diff(xp)**2 + np.diff(yp)**2)
smooth_jump = gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary
xn, yn = xp[:-1], yp[:-1]
xn = xn[(jump > 0) & (smooth_jump < limit)]
yn = yn[(jump > 0) & (smooth_jump < limit)]

pts_2D = np.array([x, y])
pts_3D = np.array([x, y, z])
pts_2D = np.array([xn, yn])
tck, u = splprep(pts_3D, u=None, s=1.5, per=1, k=2)
u_new = np.linspace(u.min(), u.max(), 10000)
x_new, y_new, z_new = splev(u_new, tck, der=0)
pts_2D = np.c_[x_new, y_new]

# plt.plot(xp, yp, '-o')
# plt.show()
plt.scatter(x, y, s=7., color='red')
plt.plot(x_new, y_new, '-o')
plt.show()

df = px.data.gapminder().query("country=='Brazil'")
fig = px.line_3d(df, x="gdpPercap", y="pop", z="year")
fig.show()




def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing