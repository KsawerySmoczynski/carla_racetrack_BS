from abc import ABCMeta, abstractmethod
import numpy as np

class Controller(metaclass=ABCMeta):
    @abstractmethod
    def control(self, state):
        pass

    @staticmethod
    def _calc_closest_dists_and_location(actor_location_3D:np.array, pts_3D:np.array):

        # Calculates
        # Take one in 10 or one in 5 points
        # pruned = pts_3D[::5,:]
        # dists = np.linalg.norm(pruned - actor_location_3D, axis=1) # -> change indexing

        dists = np.linalg.norm(pts_3D - actor_location_3D, axis=1)
        which_closest = np.argmin(dists)
        # calc distance to finish from len(waypoints)/n next
        # Length of the track / num points

        return which_closest, dists, actor_location_3D

