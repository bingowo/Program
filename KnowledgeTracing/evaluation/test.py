import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import torch 
from Constant import Constants as C
import matplotlib.pyplot as plt

class knowledge_proficiency():
    def __init__(self, model, Q, input, ) -> None:
        self.student_num = input.shape[0]
        self.knowledge_num = Q.shape[1]
        self.problem_num = Q.shape[0]
        self.result = np.zeros((self.student_num,self.knowledge_num))

        tot_know = Q.sum(0).reshape((1, self.knowledge_num))
        print(tot_know)

        for index in range(self.student_num):
            batch = torch.FloatTensor(input[index].tolist())
            batch = torch.unsqueeze(batch, dim=0).to(C.device)
            pred = model(batch).cpu()
            pred = pred[0, C.MAX_STEP-1].detach().numpy().reshape((1, self.problem_num))
            result = pred.dot(Q)
            
            # print(tot_know.shape, tot_know)
            self.result[index] = result / tot_know
        
        for i in range(self.knowledge_num):
            x = self.result[:,i]
            y = np.argsort(x)
            for j in range(self.student_num):
                self.result[y[j]][i] = j / self.student_num * 100.0

    
    def show_student(self, index):
        plt.clf()
        for i in range(C.knowledge_n-5,C.knowledge_n):
            plt.bar(C.knowledge_name[i], self.result[index,i])
        plt.ylim(0,100)
        plt.title("student:"+str(index))
        plt.savefig(str(index)+".jpg")

def show_student(index, model, figure=False):  #195
    batch = torch.FloatTensor(C.Data.result[index].tolist())
    batch = torch.unsqueeze(batch, dim=0)
    batch = batch.to(C.device)
    pred = model(batch).cpu()
    pred = pred[0, C.MAX_STEP-1].detach().numpy().reshape((1, C.NUM_OF_QUESTIONS))
    # print(pred.shape, C.Data.Q.shape)
    Q = pred.dot(C.Data.Q[1:])
    # print(Q.shape, Q)
    tot_know = C.Data.Q[1:].sum(0).reshape((1, C.knowledge_n))
    # print(tot_know.shape, tot_know)

    result = Q / tot_know
    print(result.shape, result)

    print(result)

    if figure:
        plt.clf()
        for i in range(C.knowledge_n-5,C.knowledge_n):
            plt.bar(C.knowledge_name[i], result[0,i])
        plt.ylim(0,0.5)
        plt.title("student:"+str(index))
        plt.savefig(str(index)+".jpg")

