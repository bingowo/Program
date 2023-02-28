import numpy as np
from torch.utils.data.dataset import Dataset
from Constant import Constants as C
import torch

import itertools

class DKTDataSet(Dataset):
    def __init__(self, ques, ans, paths):
        self.ques = ques
        self.ans = ans
        self.paths = paths
        self.Q = np.zeros((C.exer_n + 1, C.knowledge_n))

        with open(C.Dpath + '/contest513/problem_label.csv', 'r') as train:
            for problem in itertools.zip_longest(*[train]):
                problem_id, concepts = np.array(problem[0].strip().strip(',').split(',')).astype(np.int)
                #print(problem_id, concepts)
                num = 1
                for j in range(C.knowledge_n):
                    if (num&concepts):
                        self.Q[problem_id][j - 1] = 1
                    num = num * 2
        # print(self.Q,self.stu)
        # print(len(self.ques),len(self.ans),len(self.stu))

        self.result = np.zeros(shape=[len(self.ques), C.MAX_STEP, 2 * C.NUM_OF_QUESTIONS + (C.knowledge_n if C.use_knowledge else 0) + 3*C.MAX_CODE_LEN])
        for index in range(len(self.ques)):
            self.result[index] = self.onehot(self.ques[index], self.ans[index], self.paths[index])

        

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        # questions = self.ques[index]
        # answers = self.ans[index]
        # onehot = self.onehot(questions, answers)
        return torch.FloatTensor(self.result[index].tolist())
        #return questions, self.Q[questions] * 1.0, answers

    def onehot(self, questions, answers, paths):
        result = np.zeros(shape=[C.MAX_STEP, 2 * C.NUM_OF_QUESTIONS + (C.knowledge_n if C.use_knowledge else 0) + 3*C.MAX_CODE_LEN])
        for i in range(C.MAX_STEP):
            if answers[i] == 100:
                result[i][questions[i]] = 1
            elif answers[i] >= 0:
                result[i][questions[i] + C.NUM_OF_QUESTIONS] = 1

            if C.use_knowledge and questions[i] != -1: result[i][2*C.NUM_OF_QUESTIONS:2 * C.NUM_OF_QUESTIONS + (C.knowledge_n if C.use_knowledge else 0)] = self.Q[questions[i]]
            
            result[i][2 * C.NUM_OF_QUESTIONS + (C.knowledge_n if C.use_knowledge else 0):] = paths[i]
            
        return result