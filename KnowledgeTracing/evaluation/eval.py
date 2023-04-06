import sys
import os
import tqdm
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from ..Constant import Constants as C
from .utils import save_config_file, save_checkpoint

tt = 0
ep = 0

def performance(ground_truth, prediction, epoch, model=None):
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
    if cnt/len(ground_truth) > tt and epoch != 0:
        ep = epoch
        tt = cnt/len(ground_truth)
        # torch.save(model.state_dict(),'net_params.pth')
    if epoch != 0: print('acc:' + str(tt) + ' epoch in:' + str(ep)  + '\n')

    # if epoch == C.EPOCH-1 or (epoch > 30 and (cnt/len(ground_truth) > 0.85 or cnt/len(ground_truth) < 0.70 or auc < 0.6)):
    #     with open('is_finish.txt', 'w') as x:
    #         x.write('1')
    #     with open('result.txt', 'a') as x:
    #         x.write('acc:' + str(tt) + ' epoch in:' + str(ep)  + ' cc:' + str(cnt/len(ground_truth))  + ' auc:' + str(auc)  + '\n')
    #     exit(0)

from EOJDataset.AST.tree_tools import tree_VE_to_tensor
from EOJDataset.AST.tree_tools import collate_tree_tensor
def train_epoch(models, trainLoader, optimizers, loss_func):
    model, ast_model, trees, cache_word2vec = models
    model.train()
    ast_model.train()

    optimizer, ast_optimizer = optimizers

    loss_sum = 0
    temp_pred = torch.Tensor([]).to(C.device)
    temp_gold = torch.Tensor([]).to(C.device)
    for ques, concepts, ans, codes, score in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        ques = ques.to(C.device)                # [batch_size, MAX_STEP]
        concepts = concepts.to(C.device)        # [batch_size, MAX_STEP, 4]
        ans = ans.to(C.device)                  # [batch_size, MAX_STEP]
        # codes = codes.to(C.device)              # [batch_size, MAX_STEP]   list type
        score = score.to(C.device)              # [batch_size, MAX_STEP]


        features = torch.zeros(ques.shape[0], C.MAX_STEP, C.code_length)        
        for bs in range(ques.shape[0]):
            batch = []
            for i in range(ques.shape[1]):
                if int(codes[bs,i]) != 0:
                    nodes, mask = tree_VE_to_tensor(trees[str(int(codes[bs,i]))], cache_word2vec)
                    batch.append((nodes, mask))
            node_batch, mask_batch = collate_tree_tensor(batch)
            node_batch = node_batch.to(C.device)
            mask_batch = mask_batch.to(C.device)
            # print(node_batch.shape,mask_batch.shape)
            tmp = ast_model(node_batch, mask_batch)   # [nodes_num,local_batch_size,code_length]
            # print(tmp.shape)
            cnt = 0
            for i in range(ques.shape[1]):
                if int(codes[bs,i]) != 0:
                    features[bs,i,:] = tmp[0,cnt,:]
                    cnt += 1
        
        features = features.to(C.device)
        pred = model(ques,concepts,ans,features)   # [batch_size, MAX_STEP, 1]

        # print(pred[0,:].view(-1),'111111111111111111')
        # print(score[0,:].view(-1),'2222222222')
        # print(ans[0,:].view(-1),'333333333')

        # pred = pred.reshape(C.BATCH_SIZE,-1)
        # print(pred.size(),'======',score.size())
        # print(pred,ans)
        a = torch.tensor([]).to(C.device)
        b = torch.tensor([]).to(C.device)
        t = torch.count_nonzero(ques, dim=1).to(C.device)
        for student in range(pred.shape[0]):
            a = torch.cat([a,pred[student,1:int(t[student])]])
            b = torch.cat([b,score[student,1:int(t[student])]])
            # temp_pred = torch.cat([temp_pred, pred[student,1:int(t[student])]])
            # temp_gold = torch.cat([temp_gold, score[student,1:int(t[student])]])

        if a.shape == (0,1): continue
        
        loss = loss_func(a.view(-1), b.view(-1))
        # loss = loss_func(pred.view(-1), score.view(-1))


        # loss = torch.Tensor([0.0]).to(C.device)
        # for ind in range(ans.size()[0]):
        #     for i in range(ans.size()[1]):
        #         loss = loss - (ans[ind][i]*torch.log(pred[ind,i,1]) + (1-ans[ind][i])*torch.log(1-pred[ind,i,1]))

        optimizer.zero_grad()
        ast_optimizer.zero_grad()
        loss.backward()
        loss_sum += loss.item()
        optimizer.step()
        ast_optimizer.step()
    print('loss:',loss_sum)
    # performance(temp_gold.cpu(), temp_pred.cpu(), 0)
    return (model, ast_model, trees, cache_word2vec), (optimizer, ast_optimizer)

@torch.no_grad()
def test_epoch(models, testLoader):
    model, ast_model, trees, cache_word2vec = models
    model.eval()
    ast_model.eval()

    gold_epoch = torch.Tensor([])
    pred_epoch = torch.Tensor([])
    for ques, concepts, ans, codes, score in tqdm.tqdm(testLoader, desc='Testing:    ', mininterval=2):
        ques = ques.to(C.device)                # [batch_size, MAX_STEP]
        concepts = concepts.to(C.device)        # [batch_size, MAX_STEP, knowledge_n]
        ans = ans.to(C.device)                  # [batch_size, MAX_STEP]
        # codes = codes.to(C.device)              # [batch_size, MAX_STEP, code_length]
        # score = score.to(C.device)              # [batch_size, MAX_STEP]


        features = torch.zeros(ques.shape[0], C.MAX_STEP, C.code_length)        
        for bs in range(ques.shape[0]):
            batch = []
            for i in range(ques.shape[1]):
                if int(codes[bs,i]) != 0:
                    nodes, mask = tree_VE_to_tensor(trees[str(int(codes[bs,i]))], cache_word2vec)
                    batch.append((nodes, mask))
            node_batch, mask_batch = collate_tree_tensor(batch)
            node_batch = node_batch.to(C.device)
            mask_batch = mask_batch.to(C.device)
            # print(node_batch.shape,mask_batch.shape)
            tmp = ast_model(node_batch, mask_batch)   # [nodes_num,local_batch_size,code_length]
            # print(tmp.shape)
            cnt = 0
            for i in range(ques.shape[1]):
                if int(codes[bs,i]) != 0:
                    features[bs,i,:] = tmp[0,cnt,:]
                    cnt += 1
        
        features = features.to(C.device)
        pred = model(ques,concepts,ans,features).cpu()   # [batch_size, MAX_STEP, 1]
        # print(pred.shape,pred)


        temp_pred = torch.Tensor([])
        temp_gold = torch.Tensor([])
        t = torch.count_nonzero(ques, dim=1).cpu()
        for student in range(pred.shape[0]):
            temp_pred = torch.cat([temp_pred, pred[student,1:int(t[student])]])
            temp_gold = torch.cat([temp_gold, score[student,1:int(t[student])]])

        pred_epoch = torch.cat([pred_epoch, temp_pred])
        gold_epoch = torch.cat([gold_epoch, temp_gold])

    return pred_epoch, gold_epoch

def test(testLoader, model, epoch):
    prediction, ground_truth = test_epoch(model, testLoader)
    performance(ground_truth, prediction, epoch, model)
    print("================================================")

    