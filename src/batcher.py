import pandas as pd
import random
from observation import Observation

class Batcher:
    def __init__(self, data_dir_prefix="../data/experiments/", random_seed, BATCH_SIZE, BUFFER_SIZE):
        self.batch_size = BATCH_SIZE
        self.buffer_size = BUFFER_SIZE
        self.data_dir_prefix = data_dir_prefix
        self.all_steps = []
        self.batch = []
        self.known_csvs = []
        self.batch_iterator = BATCH_SIZE
        random.seed(random_seed)
        
        #initially explore the data system in search of already generated data
        for filename in glob.iglob(self.data_dir_prefix+'**/*.csv', recursive=True):
            self.known_csvs.append(filename)
        
        for csv_file in self.known_csvs:
            df = pd.read_csv(csv_file)
            steps = df['step'].tolist()
            for step in steps:
                self.all_steps.append(csv_file+"-"+step)
                
        self.buffer = self.all_steps[0:BATCH_SIZE]
                
    def __fill_batch():
        self.batch = []
        for i in range self.batch_size:
            drawn_step = random.choice(self.buffer).split('-')
            step = Observation(drawn_step[0], drawn_step[1])
            self.batch.append(step)

    def get_batch(new_step_csv, new_step_idx):
        self.__fill_batch()
        self.all_steps.append(new_step_csv+"-"+new_step_idx)
        self.buffer.remove(self.buffer[0])
        self.buffer.append(self.all_steps[self.batch_iterator])
        return self.batch
    
    
        