import os
import sys
sys.path.append(['..'])

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

data = [x for x in os.listdir('data/spawn_points') if '.csv' in x]
track_DF = pd.read_csv(f'data/spawn_points/{data[1]}')

# https://stackoverflow.com/questions/47948453/scipy-interpolate-splprep-error-invalid-inputs
x = track_DF['x'].values
y = track_DF['y'].values
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
pts_2D = np.array([xn, yn])
tck, u = splprep(pts_2D, u=None, s=1.5, per=1, k=2)
u_new = np.linspace(u.min(), u.max(), 1000)
x_new, y_new = splev(u_new, tck, der=0)
pts_2D = np.c_[x_new, y_new]

plt.plot(xp, yp, '-o')
plt.show()


plt.scatter(xp, yp, s=7., color='red')
plt.plot(x_new, y_new, '-o')
plt.show()
