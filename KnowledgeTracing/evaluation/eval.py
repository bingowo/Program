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

tt = 0
ep = 0

def performance(ground_truth, prediction, epoch):
    print(len(prediction))
    print(ground_truth)
    print(prediction)
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
    
    global ep
    global tt
    if cnt/len(ground_truth) > tt:
        ep = epoch
        tt = cnt/len(ground_truth)
    print('acc:' + str(tt) + ' epoch in:' + str(ep)  + '\n')


def train_epoch(model, trainLoader, optimizer, loss_func):
    loss_sum = 0
    for ques, concepts, ans, codes, score in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        ques = ques.to(C.device)                # [batch_size, MAX_STEP]
        concepts = concepts.to(C.device)        # [batch_size, MAX_STEP, knowledge_n]
        ans = ans.to(C.device)                  # [batch_size, MAX_STEP]
        codes = codes.to(C.device)              # [batch_size, MAX_STEP, code_length]
        score = score.to(C.device)              # [batch_size, MAX_STEP]

        pred = model(ques,concepts,ans,codes)   # [batch_size, MAX_STEP, 1]

        # print(pred[0,:].view(-1),'111111111111111111')
        # print(score[0,:].view(-1),'2222222222')
        # print(ans[0,:].view(-1),'333333333')

        # pred = pred.reshape(C.BATCH_SIZE,-1)
        # print(pred.size(),'======',score.size())
        # print(pred,ans)
        # a = torch.tensor([]).to(C.device)
        # b = torch.tensor([]).to(C.device)
        # t = torch.count_nonzero(ques, dim=1).to(C.device)
        # for student in range(pred.shape[0]):
        #     a = torch.cat([a,pred[student,:int(t[student])]])
        #     b = torch.cat([b,score[student,:int(t[student])]])
        # loss = loss_func(a.view(-1), b.view(-1))
        loss = loss_func(pred.view(-1), score.view(-1))

        # loss = torch.Tensor([0.0]).to(C.device)
        # for ind in range(ans.size()[0]):
        #     for i in range(ans.size()[1]):
        #         loss = loss - (ans[ind][i]*torch.log(pred[ind,i,1]) + (1-ans[ind][i])*torch.log(1-pred[ind,i,1]))

        optimizer.zero_grad()
        loss.backward()
        loss_sum += loss.item()
        # print(loss)
        optimizer.step()
    print('loss:',loss_sum)
    return model, optimizer

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

@torch.no_grad()
def test_epoch(model, testLoader): 
    gold_epoch = torch.Tensor([])
    pred_epoch = torch.Tensor([])
    for ques, concepts, ans, codes, score in tqdm.tqdm(testLoader, desc='Testing:    ', mininterval=2):
        ques = ques.to(C.device)                # [batch_size, MAX_STEP]
        concepts = concepts.to(C.device)        # [batch_size, MAX_STEP, knowledge_n]
        ans = ans.to(C.device)                  # [batch_size, MAX_STEP]
        codes = codes.to(C.device)              # [batch_size, MAX_STEP, code_length]
        # score = score.to(C.device)              # [batch_size, MAX_STEP]

        pred = model(ques,concepts,ans,codes).cpu()
        # print(pred.shape,pred)


        temp_pred = torch.Tensor([])
        temp_gold = torch.Tensor([])
        for student in range(pred.shape[0]):
            # x = int(ind[student])
            # temp_pred = torch.cat([temp_pred, pred[student,x:x+1,0]])
            # temp_gold = torch.cat([temp_gold, score[student,x:x+1]])
            for x in range(pred.shape[1]):
                if ques[student,x] != 0 and x != 0:#and (x == 49 or ques[student,x+1] == 0):
                    temp_pred = torch.cat([temp_pred, pred[student,x:x+1,0]])
                    temp_gold = torch.cat([temp_gold, score[student,x:x+1]])

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

    