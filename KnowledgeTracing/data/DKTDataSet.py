import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from ..Constant import Constants as C
from ..data.readdata import DataReader
from EOJDataset.AST.tree_tools import tree_VE_to_tensor
from EOJDataset.AST.word2vec import create_word_dict

import itertools
import pandas as pd

class DKTDataSet(Dataset):
    def __init__(self, data_path, use_data_augmentation=False):
        handle = DataReader(data_path, C.MAX_STEP)
        ques, ans, code_ids = handle.getData(use_data_augmentation)

        self.ques = ques
        self.ans = ans
        self.code_ids = code_ids

        self.Q = np.zeros((C.exer_n + 1, 4))
        prob2concept = pd.read_csv(C.Dpath + '/contest513/problem_label.csv')
        for i in range(len(prob2concept)):
            problem_id, concepts = prob2concept.loc[i,'problem_id'], prob2concept.loc[i,'concept']
            num = 1
            cnt = 0
            for j in range(C.knowledge_n):
                if (num&concepts):
                    # self.Q[problem_id][j] = 1
                    self.Q[problem_id][cnt] = j + 1
                    cnt += 1
                num = num * 2
        self.Q = torch.tensor(self.Q)

        # print(self.Q,self.stu)
        # print(len(self.ques),len(self.ans),len(self.stu))

        # codes = torch.load('EOJDataset/CodeBERT/code_embeddings_codebert.t7')
        codes = None

        self.in_ques        = torch.zeros([len(self.ques), C.MAX_STEP])
        self.in_concepts    = torch.zeros([len(self.ques), C.MAX_STEP, 4])
        self.in_score         = torch.zeros([len(self.ques), C.MAX_STEP])
        self.in_codes       = torch.zeros([len(self.ques), C.MAX_STEP, C.code_length])
        for index in range(len(self.ques)):
            self.in_ques[index,:], self.in_concepts[index,:], self.in_score[index,:], self.in_codes[index,:] = self.preload(index, codes)

        # self.result = np.zeros(shape=[len(self.ques), C.MAX_STEP, 2 * C.NUM_OF_QUESTIONS + (C.knowledge_n if C.use_knowledge else 0)])
        # for index in range(len(self.ques)):
        #     self.result[index] = self.onehot(self.ques[index], self.ans[index])

        self.out_ans        = torch.zeros([len(self.ques), C.MAX_STEP])
        self.out_codes      = torch.zeros([len(self.ques), C.MAX_STEP, C.code_length])

        self.out_ans[:,1:]     = self.in_score[:,:-1]
        self.out_codes[:,1:]   = self.in_codes[:,:-1]
        
        self.in_ques = self.in_ques.long()
        self.in_concepts = self.in_concepts.long()
        self.out_ans = self.out_ans.long()
        # self.out_codes = self.out_codes.long()
        # self.output = self.output.long()

        

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        return self.in_ques[index], self.in_concepts[index], self.out_ans[index], self.out_codes[index], self.in_score[index]
        #return questions, self.Q[questions] * 1.0, answers

    def preload(self, index, code2):
        ques    = torch.from_numpy(self.ques[index])
        concept = torch.zeros([C.MAX_STEP, 4])
        ans     = torch.tensor([1 if self.ans[index][i] == 100 else 0 for i in range(C.MAX_STEP)],dtype=torch.long)
        codes   = torch.zeros([C.MAX_STEP, C.code_length])
        for i in range(C.MAX_STEP):
            # print(self.nums[index,i],self.vecs.get(self.nums[index,i],torch.zeros((1,768))).size())
            concept[i,:] = self.Q[self.ques[index][i],:]
            # codes[i,:] = code2.get(str(self.code_ids[index, i]), torch.zeros((1, C.code_length)))
            # codes[i,:] = torch.nn.functional.normalize(codes[i,:], p=2, dim=0)
            # codes[i,:] = torch.zeros((1, C.code_length))
            # if codes[i,:].sum() == 0 and self.code_ids[index, i] != 0:  print(codes[i,:].sum())
        
        # c = torch.FloatTensor([self.vecs.get(self.nums[index,i],torch.zeros((1,768))) for i in range(C.MAX_STEP)])
        return ques.long(), concept, ans, codes
    
    def onehot(self, questions, answers):
        result = np.zeros(shape=[C.MAX_STEP, 2 * C.NUM_OF_QUESTIONS + (C.knowledge_n if C.use_knowledge else 0)])
        for i in range(C.MAX_STEP):
            if answers[i] == 1:
                result[i][questions[i]] = 1
            elif questions[i] > 0:
                result[i][questions[i] + C.NUM_OF_QUESTIONS] = 1

            if C.use_knowledge and questions[i] != 0: result[i][2*C.NUM_OF_QUESTIONS:] = self.Q[questions[i]]
            
        return torch.FloatTensor(result)