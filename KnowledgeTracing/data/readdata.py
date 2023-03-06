import numpy as np
from data.DKTDataSet import DKTDataSet
import itertools
import tqdm

class DataReader():
    def __init__(self, path, maxstep, numofques):
        self.path = path
        self.maxstep = maxstep
        self.numofques = numofques

    def getData(self):
        trainqus = np.array([])
        trainans = np.array([])
        traincode_id = np.array([])
        with open(self.path, 'r') as train:
            for len, ques, ans, code_id in tqdm.tqdm(itertools.zip_longest(*[train] * 4), desc='loading data:    ', mininterval=2):
                len = int(len.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(np.int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(np.int)
                code_id = np.array(code_id.strip().strip(',').split(',')).astype(np.int)
                
                mod = 0 if len%self.maxstep == 0 else (self.maxstep - len%self.maxstep)
                zero = np.zeros(mod) #- 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                code_id = np.append(code_id, zero)
                trainqus = np.append(trainqus, ques).astype(np.int)
                trainans = np.append(trainans, ans).astype(np.int)
                traincode_id = np.append(traincode_id, code_id).astype(np.int)
        return trainqus.reshape([-1, self.maxstep]), trainans.reshape([-1, self.maxstep]), traincode_id.reshape([-1, self.maxstep])