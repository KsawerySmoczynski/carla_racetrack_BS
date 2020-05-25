import pandas as pd

class Observation:
    """Class storing data about a single step"""
    def __init__(self, observation_filename = "../data/experiments/", step_id = 0):
        self.camera_data_dir = observation_filename+'sensors/'
        df = pd.read_csv(observation_filename)
        raw_obs_data = df[df['step']==step_id].values.tolist()[0]
        self.step, self.depth_camera_files, self.rgb_camera_files, self.collisions, self.velocity, self.velocity_cev, self.yaw, \
        self.location, self.distance_2finish, self.steer, self.gas_break, self.reward = raw_obs_data
        self.depth_camera_data = extract_camera_data(self.depth_camera_files,'depth')
        self.rgb_camera_data = extract_camera_data(self.rgb_camera_files, 'rgb')
        
    def __extract_camera_data(file_indices = [0, 1, 2, 3], camera_type = 'depth'):
        camera_data = []
        for index in file_indices:
            camera_data.append(np.array(Image.open(self.camera_data_dir+'{}_{}.png'.format(camera_type, str(index)))).astype(np.uint8)) 
        return camera_data
        