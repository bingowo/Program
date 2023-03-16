import torch
import tqdm

from ..Constant import Constants as C

def train_epoch(model, trainLoader, optimizer, loss_func):
    model, classifier = model
    model.train()
    classifier.train()

    loss_sum = 0
    top1 = 0
    top5 = 0
    tot = 0
    for ques, ans, nodes, mask in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        ques = ques.to(C.device)                # [batch_size]
        ans = ans.to(C.device)                  # [batch_size, MAX_STEP]
        nodes = nodes.to(C.device)              # [length, batch_size, code_length = 384]
        mask = mask.to(C.device)              # [batch_size, length, length]

        hidden = model(nodes,mask)[0]   # [batch_size, hidden_size]
        output = classifier(hidden) # [batch_size, exer_n]

        # print(ques.shape,ans.shape,nodes.shape,mask.shape,output.shape)
    
        loss = loss_func(output, ques)

        tot += ques.shape[0]
        _, ind = torch.topk(output, 5)
        for i in range(ques.shape[0]):
            # print(ques[i],ind[i])
            if ques[i] in ind[i]:
                top5 += 1
            if ques[i] == ind[i,0]:
                top1 += 1

        optimizer.zero_grad()
        loss.backward()
        loss_sum += loss.item()
        # print(loss)
        optimizer.step()
    print('loss: {}, top1: {}, top5: {}'.format(loss_sum,top1/tot*100,top5/tot*100))
    

    return (model, classifier), optimizer


@torch.no_grad()
def test_epoch(model, testLoader): 
    model, classifier = model
    model.eval()
    classifier.eval()

    top1 = 0
    top5 = 0
    tot = 0
    for ques, ans, nodes, mask in tqdm.tqdm(testLoader, desc='Testing:    ', mininterval=2):
        ques = ques.to(C.device)                # [batch_size]
        ans = ans.to(C.device)                  # [batch_size, MAX_STEP]
        nodes = nodes.to(C.device)              # [length, batch_size, code_length = 384]
        mask = mask.to(C.device)              # [batch_size, length, length]

        hidden = model(nodes,mask)[0]   # [batch_size, hidden_size]
        output = classifier(hidden) # [batch_size, exer_n]

        tot += ques.shape[0]
        _, ind = torch.topk(output, 5)
        for i in range(ques.shape[0]):
            # print(ques[i],ind[i])
            if ques[i] in ind[i]:
                top5 += 1
            if ques[i] == ind[i,0]:
                top1 += 1
    print('top1: {}, top5: {}'.format(top1/tot*100,top5/tot*100))
    return

def test(testLoader, model, epoch):
    test_epoch(model, testLoader)
    print("================================================")