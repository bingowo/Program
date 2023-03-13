import numpy as np
from data.DKTDataSet import DKTDataSet
import itertools
import tqdm
import random

class DataReader():
    def __init__(self, path, maxstep, numofques):
        self.path = path
        self.maxstep = maxstep
        self.numofques = numofques

    def getData(self, resi):
        trainqus = np.array([])
        trainans = np.array([])
        traincode_id = np.array([])
        end = 50 if resi else 1
        with open(self.path, 'r') as train:
            for len, ques, ans, code_id in tqdm.tqdm(itertools.zip_longest(*[train] * 4), desc='loading data:    ', mininterval=2):
                _len = int(len.strip().strip(','))
                _ques = np.array(ques.strip().strip(',').split(',')).astype(np.int)
                _ans = np.array(ans.strip().strip(',').split(',')).astype(np.int)
                _code_id = np.array(code_id.strip().strip(',').split(',')).astype(np.int)
                for i in range(0,end,10):
                    if _len > i:
                        len = _len - i
                        ques = _ques[i:]
                        ans = _ans[i:]
                        code_id = _code_id[i:]
                        
                        mod = 0 if len%self.maxstep == 0 else (self.maxstep - len%self.maxstep)
                        zero = np.zeros(mod) #- 1
                        ques = np.append(ques, zero)
                        ans = np.append(ans, zero)
                        code_id = np.append(code_id, zero)
                        trainqus = np.append(trainqus, ques).astype(np.int)
                        trainans = np.append(trainans, ans).astype(np.int)
                        traincode_id = np.append(traincode_id, code_id).astype(np.int)
                if resi:
                    for _ in range((_len-80)//5):
                        x = random.sample(range(_len), 50)
                        x.sort()
                        len = 50
                        ques = [_ques[i] for i in x]
                        ans = [_ans[i] for i in x]
                        code_id = [_code_id[i] for i in x]
                        
                        mod = 0 if len%self.maxstep == 0 else (self.maxstep - len%self.maxstep)
                        zero = np.zeros(mod) #- 1
                        ques = np.append(ques, zero)
                        ans = np.append(ans, zero)
                        code_id = np.append(code_id, zero)
                        trainqus = np.append(trainqus, ques).astype(np.int)
                        trainans = np.append(trainans, ans).astype(np.int)
                        traincode_id = np.append(traincode_id, code_id).astype(np.int)
        print("data number: ",trainqus.size//50)
        return trainqus.reshape([-1, self.maxstep]), trainans.reshape([-1, self.maxstep]), traincode_id.reshape([-1, self.maxstep])