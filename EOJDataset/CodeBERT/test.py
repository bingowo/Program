import torch
import itertools as it
import random

# codes = torch.load('EOJDataset/CodeBERT/code_embeddings.t7')
# codes = torch.load('EOJDataset/CodeBERT/code_embeddings_codebert.t7')
# print(len(codes))

# i=2865921
# j=2866071
# x = codes[str(i)].reshape(1,-1)
# y = codes[str(j)].reshape(1,-1)
# print(x.shape,x.max(),x.min())
# print(y.shape,y.max(),y.min())
# x = torch.nn.functional.normalize(x, p=2, dim=1)
# y = torch.nn.functional.normalize(y, p=2, dim=1)
# # print(x.shape,x.max(),x.min(),x)
# # print(y.shape,y.max(),y.min(),y)
# t = torch.einsum("ac,bc->ab",x,y)
# print(t)

for i in range(20):
    x = random.sample(range(10), 5)
    x.sort()
    print(x)