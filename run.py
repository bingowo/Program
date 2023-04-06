import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from KnowledgeTracing.model.RNNModel import DKT
from KnowledgeTracing.model.SAINTModel import saint
from KnowledgeTracing.model.ast_attention import AstAttention
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.evaluation import eval, ast_eval
from KnowledgeTracing.evaluation.test import knowledge_proficiency
from KnowledgeTracing.data.DKTDataSet import DKTDataSet
from KnowledgeTracing.data.ASTDataSet import ASTDataSet

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

a = 0.0002
b = 0.3
c = 0.0002

model = saint(dim_model=128,
            num_en=2,
            num_de=2,
            heads_en=8,
            heads_de=8,
            total_ex=C.exer_n,
            total_cat=C.knowledge_n+1,
            total_in=2,
            seq_len=C.MAX_STEP,
            dropout=b
            ).to(C.device)
ast_model = AstAttention(384, C.code_length, num_layers=2, num_heads=8).to(C.device)
# from KnowledgeTracing.model.classifier import Classifier
# classifier = Classifier(768, C.exer_n).to(C.device)
# model = DKT(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT).to(C.device)
# model = NeuralCDM(C.student_n, C.exer_n, C.knowledge_n).to(C.device)
# net = NeuralCDM(C.student_n, C.exer_n, C.knowledge_n)
# loss_func = nn.CrossEntropyLoss().to(C.device)
loss_func = nn.BCELoss().to(C.device)
# loss_func = nn.MSELoss().to(C.device)
# loss_func = eval.lossFunc().to(C.device)
#loss_func = nn.NLLLoss().to(C.device)

# optimizer = torch.optim.AdamW([
# 	{"params": model.parameters(), "lr": 1e-4, "weight_decay": 0.1}, 
# 	{"params": classifier.parameters(), "lr": 3e-4}
# ])
optimizer = optim.Adam(model.parameters(), lr=a, weight_decay=c)#, weight_decay=1e-4)
ast_optimizer = optim.Adam(ast_model.parameters(), lr=a, weight_decay=c)
optimizer = (optimizer, ast_optimizer)
# optimizer = optim.Adagrad(model.parameters(),lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=C.LR, weight_decay=1e-4, momentum=0.98)

total = sum(p.numel() for p in model.parameters())
print("Model params: %.2fM" % (total/1e6))
total = sum(p.numel() for p in ast_model.parameters())
print("AST_model params: %.2fM" % (total/1e6))

import json
from EOJDataset.AST.word2vec import create_word_dict
with open("EOJDataset/AST/ast_trees_pruned.json",'r') as f:
    _trees = json.load(f)
_cache_word2vec = create_word_dict(trees=_trees)
print("Loaded trees and word_cache.")
model = (model,ast_model,_trees,_cache_word2vec)

train_ddata = DKTDataSet(C.Dpath + '/contest513/contest513_train.csv', use_data_augmentation=False)
test_ddata = DKTDataSet(C.Dpath + '/contest513/contest513_test.csv')
# from EOJDataset.AST.tree_tools import collate_tree_tensor
# train_ddata = ASTDataSet(C.Dpath + '/contest513/contest513_train.csv')
# test_ddata = ASTDataSet(C.Dpath + '/contest513/contest513_test.csv')
trainLoader = Data.DataLoader(train_ddata, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=C.WORKERS)
testLoader = Data.DataLoader(test_ddata, batch_size=C.BATCH_SIZE, num_workers=C.WORKERS)


# self.writer = SummaryWriter()
# logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

epoch_begin = 0

# tmp = torch.load('tmp.pth')
# epoch_begin = tmp['epoch'] + 1
# model[0].load_state_dict(tmp['model'])
# model[1].load_state_dict(tmp['ast_model'])
# optimizer[0].load_state_dict(tmp['opt'])
# optimizer[1].load_state_dict(tmp['ast_opt'])

for epoch in range(epoch_begin, C.EPOCH):
    print('epoch: ' + str(epoch))
    model, optimizer = eval.train_epoch(model, trainLoader, optimizer, loss_func)
    torch.save({'epoch':epoch,'model':model[0].state_dict(),'ast_model':model[1].state_dict(),'opt':optimizer[0].state_dict(),'ast_opt':optimizer[1].state_dict()},'tmp.pth')
    # eval.validate_for_NeuralCDM(trainLoaders, model, epoch, net)
    eval.test(testLoader, model, epoch)

# torch.save(model.state_dict(),'net_params.pth')

# state_dict=torch.load('net_params.pth')
# model.load_state_dict(state_dict)

# knowledge_master =  knowledge_proficiency(model, C.Data.Q[1:], C.Data.result)
# for i in range(0, C.student_n):
#     knowledge_master.show_student(i)

# for i in range(0, C.student_n):
#     print("student",i,end=": ")
#     test.show_student(i, model, figure=True)