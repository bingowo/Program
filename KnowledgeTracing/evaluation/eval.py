import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tqdm
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from Constant import Constants as C
from utils import save_config_file, save_checkpoint

def performance(ground_truth, prediction, epoch):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().numpy(), prediction.detach().numpy())
    auc = metrics.auc(fpr, tpr)

    f1 = metrics.f1_score(ground_truth.detach().numpy(), torch.round(prediction).detach().numpy())
    recall = metrics.recall_score(ground_truth.detach().numpy(), torch.round(prediction).detach().numpy())
    precision = metrics.precision_score(ground_truth.detach().numpy(), torch.round(prediction).detach().numpy())

    cnt = 0
    for i in range(len(ground_truth)):
        if prediction[i] > 0.5:
            if int(ground_truth[i]) == 1:
                cnt = cnt + 1
        else:
            if int(ground_truth[i]) == 0:
                cnt = cnt + 1

    print('acc:' + str(cnt/len(ground_truth)) + ' auc:' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) + ' precision: ' + str(precision) + '\n')
    with open('result/model_test.txt', 'a', encoding='utf8') as f:
        f.write('epoch:' + str(epoch) + ' acc:' + str(cnt/len(ground_truth)) + ' auc:' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) + ' precision: ' + str(precision) + '\n')


class lossFunc(nn.Module):
    def __init__(self):
        super(lossFunc, self).__init__()

    def forward(self, pred, batch):
        loss = torch.Tensor([0.0]).to(C.device)
        for student in range(pred.shape[0]):
            delta = batch[student][:,0:C.NUM_OF_QUESTIONS] + batch[student][:,C.NUM_OF_QUESTIONS:]
            temp = pred[student][:C.MAX_STEP - 1].mm(delta[1:].t())
            index = torch.LongTensor([[i for i in range(C.MAX_STEP - 1)]]).to(C.device)
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:C.NUM_OF_QUESTIONS] - batch[student][:, C.NUM_OF_QUESTIONS:]).sum(1) + 1)//2)[1:]
            
            for i in range(len(p)):
                if p[i] > 0:
                    loss = loss - (a[i]*torch.log(p[i]) + (1-a[i])*torch.log(1-p[i]))
        return loss

def train_epoch(model, trainLoader, optimizer, loss_func):
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch = batch.to(C.device)
        pred = model(batch)
        batch = batch[:,:,:2*C.NUM_OF_QUESTIONS]

        loss = loss_func(pred, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, optimizer

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

def test_epoch(model, testLoader): 
    gold_epoch = torch.Tensor([])
    pred_epoch = torch.Tensor([])
    for batch in tqdm.tqdm(testLoader, desc='Testing:    ', mininterval=2):
        batch = batch.to(C.device)
        pred = model(batch)

        batch = batch[:,:,:2*C.NUM_OF_QUESTIONS]
        # pred = pred * torch.tensor(0.01).to(C.device)

        for student in range(pred.shape[0]):
            temp_pred = torch.Tensor([])
            temp_gold = torch.Tensor([])
            delta = batch[student][:,0:C.NUM_OF_QUESTIONS] + batch[student][:,C.NUM_OF_QUESTIONS:]
            temp = pred[student][:C.MAX_STEP - 1].mm(delta[1:].t()).cpu()
            index = torch.LongTensor([[i for i in range(C.MAX_STEP - 1)]])
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:C.NUM_OF_QUESTIONS] - batch[student][:, C.NUM_OF_QUESTIONS:]).sum(1) + 1)//2)[1:].cpu()
            for i in range(len(p)):
                if p[i] > 0:
                    temp_pred = torch.cat([temp_pred,p[i:i+1]])
                    temp_gold = torch.cat([temp_gold,a[i:i+1]])
            pred_epoch = torch.cat([pred_epoch, temp_pred])
            gold_epoch = torch.cat([gold_epoch, temp_gold])
    return pred_epoch, gold_epoch


def train(trainLoaders, model, optimizer, lossFunc):
    for i in range(len(trainLoaders)):
        model, optimizer = train_epoch(model, trainLoaders[i], optimizer, lossFunc)
    return model, optimizer

def test(testLoaders, model, epoch):
    ground_truth = torch.Tensor([])
    prediction = torch.Tensor([])
    for i in range(len(testLoaders)):
        pred_epoch, gold_epoch = test_epoch(model, testLoaders[i])
        prediction = torch.cat([prediction, pred_epoch])
        ground_truth = torch.cat([ground_truth, gold_epoch])
    performance(ground_truth, prediction, epoch)

    