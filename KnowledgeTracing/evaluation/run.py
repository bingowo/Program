import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from model.RNNModel import DKT
from model.NeuralCDModel import NeuralCDM
from model.c2vRNNModel import c2vRNNModel
from data.dataloader import getLoaderSet, getLoader
from Constant import Constants as C
import torch
import torch.nn as nn
import torch.optim as optim
from evaluation import eval
from evaluation.test import knowledge_proficiency

print('Dataset: ' + C.DATASET + ', Learning Rate: ' + str(C.LR) + '\n')


model = c2vRNNModel(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT, 7002, 7297240).to(C.device)

# model = DKT(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT).to(C.device)
# model = NeuralCDM(C.student_n, C.exer_n, C.knowledge_n).to(C.device)
# net = NeuralCDM(C.student_n, C.exer_n, C.knowledge_n)
loss_func = eval.lossFunc().to(C.device)
#loss_func = nn.NLLLoss().to(C.device)

# optimizer = optim.Adam(model.parameters(), lr=C.LR)
optimizer = optim.Adagrad(model.parameters(),lr=C.LR)

trainLoaders, testLoaders = getLoader(C.DATASET)


# self.writer = SummaryWriter()
# logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

state_dict=torch.load('net_params_epoch_19.pth')
model.load_state_dict(state_dict)

for epoch in range(20, C.EPOCH):
    print('epoch: ' + str(epoch))
    model, optimizer = eval.train(trainLoaders, model, optimizer, loss_func)
    # eval.validate_for_NeuralCDM(trainLoaders, model, epoch, net)
    eval.test(testLoaders, model, epoch)

    if epoch%10 == 9:
        torch.save(model.state_dict(),'net_params_epoch_'+str(epoch)+'.pth')

# knowledge_master =  knowledge_proficiency(model, C.Data.Q[1:], C.Data.result)
# for i in range(0, C.student_n):
#     knowledge_master.show_student(i)

# for i in range(0, C.student_n):
#     print("student",i,end=": ")
#     test.show_student(i, model, figure=True)