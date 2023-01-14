import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from model.RNNModel import DKT
from model.NeuralCDModel import NeuralCDM
from data.dataloader import getLoaderSet, getLoader
from Constant import Constants as C
import torch
import torch.nn as nn
import torch.optim as optim
from evaluation import eval

print('Dataset: ' + C.DATASET + ', Learning Rate: ' + str(C.LR) + '\n')

model = DKT(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT).to(C.device)
# model = NeuralCDM(C.student_n, C.exer_n, C.knowledge_n).to(C.device)
# net = NeuralCDM(C.student_n, C.exer_n, C.knowledge_n)
loss_func = eval.lossFunc().to(C.device)
#loss_func = nn.NLLLoss().to(C.device)

# optimizer = optim.Adam(model.parameters(), lr=C.LR)
optimizer = optim.Adagrad(model.parameters(),lr=C.LR)

trainLoaders, testLoaders = getLoader(C.DATASET)


# self.writer = SummaryWriter()
# logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

for epoch in range(C.EPOCH):
    print('epoch: ' + str(epoch))
    model, optimizer = eval.train(trainLoaders, model, optimizer, loss_func)
    # eval.validate_for_NeuralCDM(trainLoaders, model, epoch, net)
    eval.test(testLoaders, model, epoch)
