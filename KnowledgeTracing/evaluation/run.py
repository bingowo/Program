import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from model.RNNModel import DKT
from model.NeuralCDModel import NeuralCDM
from model.SAINTModel import saint
from data.dataloader import getLoaderSet, getLoader
from Constant import Constants as C
import torch
import torch.nn as nn
import torch.optim as optim
from evaluation import eval
from evaluation.test import knowledge_proficiency

print('Dataset: ' + C.DATASET + ', Learning Rate: ' + str(C.LR) + '\n')


# with open('is_finish.txt', 'w') as x:
#     x.write('0')
# with open('parameter.txt', 'r') as x:
#     a,b,c = x.readline().split()
#     a = float(a)
#     b = float(b)
#     c = float(c)
# with open('result.txt', 'a') as x:
#     x.write('lr:' + str(a) + ' dropout:' + str(b)  + ' weight_decay:' + str(c)  +  '    ')

a = 0.01
b = 0.1
c = 0.0002

model = saint(dim_model=128,
            num_en=1,
            num_de=1,
            heads_en=8,
            heads_de=8,
            total_ex=C.exer_n,
            total_cat=C.knowledge_n+1,
            total_in=2,
            seq_len=C.MAX_STEP,
            dropout=b
            ).to(C.device)
# model = TModel(128).to(C.device)
# model = DKT(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT).to(C.device)
# model = NeuralCDM(C.student_n, C.exer_n, C.knowledge_n).to(C.device)
# net = NeuralCDM(C.student_n, C.exer_n, C.knowledge_n)
# loss_func = nn.CrossEntropyLoss().to(C.device)
loss_func = nn.BCELoss().to(C.device)
# loss_func = nn.MSELoss().to(C.device)
# loss_func = eval.lossFunc().to(C.device)
#loss_func = nn.NLLLoss().to(C.device)

optimizer = optim.Adam(model.parameters(), lr=a, weight_decay= c)#, weight_decay=1e-4)
# optimizer = optim.Adagrad(model.parameters(),lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=C.LR, weight_decay=1e-4, momentum=0.98)

total = sum(p.numel() for p in model.parameters())
print("Total params: %.2fM" % (total/1e6))

trainLoaders, testLoaders = getLoader(C.DATASET)


# self.writer = SummaryWriter()
# logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

for epoch in range(C.EPOCH):
    print('epoch: ' + str(epoch))
    model, optimizer = eval.train(trainLoaders, model, optimizer, loss_func)
    # eval.validate_for_NeuralCDM(trainLoaders, model, epoch, net)
    eval.test(testLoaders, model, epoch)

# torch.save(model.state_dict(),'net_params.pth')

# state_dict=torch.load('net_params.pth')
# model.load_state_dict(state_dict)

# knowledge_master =  knowledge_proficiency(model, C.Data.Q[1:], C.Data.result)
# for i in range(0, C.student_n):
#     knowledge_master.show_student(i)

# for i in range(0, C.student_n):
#     print("student",i,end=": ")
#     test.show_student(i, model, figure=True)