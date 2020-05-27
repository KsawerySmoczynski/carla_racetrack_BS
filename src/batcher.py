import pandas as pd
import random
from observation import Observation

class Batcher:
    def __init__(self, data_dir_prefix="../data/experiments/", random_seed, BATCH_SIZE, BUFFER_SIZE):
        self.batch_size = BATCH_SIZE
        self.buffer_size = BUFFER_SIZE
        self.data_dir_prefix = data_dir_prefix
        self.buffer = []
        self.batch = []
        self.known_csvs = []
        self.current_step_idx = 0
        random.seed(random_seed)
        
        #initially explore the data system in search of already generated data
        for filename in glob.iglob(self.data_dir_prefix+'**/*.csv', recursive=True):
            self.known_csvs.append(filename)
        
        #problemy:
        #1. ładuję dane i jest poniżej buffersize
        #2. ładuję dane i kończę w połowie pliku
        #3. ładuję dane i kończę idealnie na końcu pliku
        
        self.current_csv_idx = 0
        df = pd.read_csv(self.known_csvs[self.current_csv_idx])
        self.current_csv_steps = df['step'].tolist()
        
        
        while len(self.buffer) < BUFFER_SIZE:
            try:
                self.append_buffer()
            except:
                #should get here if we try to open a csv when we run out of them
                print("Ran out of data to read while preloading data buffer")
     
    def __fill_batch():
        self.batch = []
        for i in range self.batch_size:
            drawn_step = random.choice(self.buffer).split('-')
            step = Observation(drawn_step[0], drawn_step[1])
            self.batch.append(step)

    def append_buffer():
        self.buffer.append(self.known_csvs[self.current_csv_idx] + "-" + self.current_step_idx)
        self.current_step_idx += 1
        if len(self.buffer) > BUFFER_SIZE:
            self.buffer.remove(self.buffer[0])
        if self.current_step_idx == len(self.current_csv_steps):
            self.current_csv_idx += 1
            df = pd.read_csv(self.known_csvs[self.current_csv_idx])
            self.current_csv_steps = df['step'].tolist()
            self.current_step_idx = 0
        
    
    def get_batch(new_step_csv, new_step_idx):
        self.__fill_batch()
        self.append_buffer()
        return self.batch
    
    
        