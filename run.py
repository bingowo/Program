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

# a = 0.0001
# b = 0.5
# c = 0.1

# model = saint(dim_model=256,
#             num_en=6,
#             num_de=6,
#             heads_en=8,
#             heads_de=8,
#             total_ex=C.exer_n,
#             total_cat=C.knowledge_n+1,
#             total_in=2,
#             seq_len=C.MAX_STEP,
#             dropout=b
#             ).to(C.device)
model = AstAttention(384, 768, num_layers=6, num_heads=8).to(C.device)
from KnowledgeTracing.model.classifier import Classifier
classifier = Classifier(768, C.exer_n).to(C.device)
# model = DKT(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT).to(C.device)
# model = NeuralCDM(C.student_n, C.exer_n, C.knowledge_n).to(C.device)
# net = NeuralCDM(C.student_n, C.exer_n, C.knowledge_n)
loss_func = nn.CrossEntropyLoss().to(C.device)
# loss_func = nn.BCELoss().to(C.device)
# loss_func = nn.MSELoss().to(C.device)
# loss_func = eval.lossFunc().to(C.device)
#loss_func = nn.NLLLoss().to(C.device)

optimizer = torch.optim.AdamW([
	{"params": model.parameters(), "lr": 1e-4, "weight_decay": 0.1}, 
	{"params": classifier.parameters(), "lr": 3e-4}
])
# optimizer = optim.AdamW(model.parameters(), lr=a, weight_decay=c)#, weight_decay=1e-4)
# optimizer = optim.Adagrad(model.parameters(),lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=C.LR, weight_decay=1e-4, momentum=0.98)

total = sum(p.numel() for p in model.parameters())
print("Total params: %.2fM" % (total/1e6))

model = (model,classifier)

# train_ddata = DKTDataSet(C.Dpath + '/contest513/contest513_train.csv', use_data_augmentation=False)
# test_ddata = DKTDataSet(C.Dpath + '/contest513/contest513_test.csv')
from EOJDataset.AST.tree_tools import collate_tree_tensor
train_ddata = ASTDataSet(C.Dpath + '/contest513/contest513_train.csv')
test_ddata = ASTDataSet(C.Dpath + '/contest513/contest513_test.csv')
trainLoader = Data.DataLoader(train_ddata, batch_size=C.BATCH_SIZE, collate_fn=collate_tree_tensor, shuffle=True, num_workers=C.WORKERS)
testLoader = Data.DataLoader(test_ddata, batch_size=C.BATCH_SIZE, collate_fn=collate_tree_tensor, num_workers=C.WORKERS)


# self.writer = SummaryWriter()
# logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

for epoch in range(C.EPOCH):
    print('epoch: ' + str(epoch))
    model, optimizer = ast_eval.train_epoch(model, trainLoader, optimizer, loss_func)
    # eval.validate_for_NeuralCDM(trainLoaders, model, epoch, net)
    ast_eval.test(testLoader, model, epoch)

# torch.save(model.state_dict(),'net_params.pth')

# state_dict=torch.load('net_params.pth')
# model.load_state_dict(state_dict)

# knowledge_master =  knowledge_proficiency(model, C.Data.Q[1:], C.Data.result)
# for i in range(0, C.student_n):
#     knowledge_master.show_student(i)

# for i in range(0, C.student_n):
#     print("student",i,end=": ")
#     test.show_student(i, model, figure=True)