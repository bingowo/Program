import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ',device)
tokenizer = RobertaTokenizer.from_pretrained("neulab/codebert-c")
model = RobertaModel.from_pretrained("neulab/codebert-c")
model.to(device)
print("model loaded.")

import os
from tqdm import tqdm
import pandas as pd

# 去除程序中的头文件
def rm_includeline(ms):
    if not isinstance(ms, str):
        raise TypeError(ms)
    ms = "".join([s for s in ms.splitlines(True) if 'include' not in s or '#' != s.lstrip()[0]])
    return ms

@torch.no_grad()
def file2vec(filename):
    number = filename.split('\\')[-1].split('_')[1][1:]
    # print(filename)

    with open(filename, encoding='utf-8') as f:
        txt = f.read()
    # txt = rm_includeline(txt)  # 去除头文件

    # txt = '#define bool int\n' + txt
    # txt = '#define uint32_t unsigned int\n' + txt
    # txt = '#define __int64_t long long\n' + txt
    # txt = '#define size_t unsigned long long \n' + txt
    # txt = '#define extern int \n' + txt
    # txt = '#define int32_t int\n' + txt

    # with open('EOJDataset/CodeBERT/tmp.c','w', encoding='utf-8') as f:
    #     f.write(txt)
    # # print(txt)

    # cmd = r'"C:\\Program Files (x86)\Dev-Cpp\\MinGW64\\bin\\gcc.exe" -E EOJDataset\\CodeBERT\\tmp.c -o EOJDataset\\CodeBERT\\test.c'
    # os.system(cmd)

    # with open('EOJDataset\\CodeBERT\\test.c', encoding='utf-8') as f:
    #     txt = f.read()
    # # txt = txt[120:]
    # while txt[0]=='#':
    #     i = 0
    #     while(txt[i]!='\n'): i += 1
    #     txt = txt[i+1:]
    # while(txt[0]=='\n'): txt = txt[1:]
    # print(filename)
    # print(txt)

    #====================================================================

    code_tokens=tokenizer.tokenize(txt)
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    tokens=tokenizer.convert_tokens_to_ids(tokens)
    # print(min(tokens),max(tokens),len(tokens))

    x = tokens[:min(len(tokens),512)]
    # print(code_tokens)
    # print(tokens)
    x = torch.tensor(x[:]).to(device)
        
    embedding=model(x[None,:])[0]
    embedding=embedding[0,0,:]
    
    return number, embedding.cpu()



# filename = 'EOJDataset/contest513/ContestCode\\10215102508_陈鑫_王老师_102092\\1106_#3026056_Accepted.c'
# number, vecs = file2vec(filename)
# print(vecs)
# print(number)

code_dir = 'EOJDataset/contest513/ContestCode'
code_files = []
for root, dirs, files in os.walk(code_dir):
    for file in files:
        if 'Compilation' not in file:
            code_files.append(os.path.join(root, file))

paths = {}
print(len(code_files))
cnt = 0
for file in tqdm(code_files,mininterval=3):
    number, path = file2vec(file)
    paths[number] = path
    cnt += 1

    if cnt % 5000 == 0:
        # main_df = pd.DataFrame.from_dict(paths, orient='index')
        # main_df.to_csv("EOJDataset/contest513/labeled_"+str(cnt)+"_paths.tsv", sep="\t", header=True)
        torch.save(paths,"EOJDataset/CodeBERT/code_embedding_"+str(cnt)+".t7")

# main_df = pd.DataFrame.from_dict(paths, orient='index')
# main_df.to_csv("EOJDataset/contest513/labeled_paths.tsv", sep="\t", header=True)
torch.save(paths,"EOJDataset/CodeBERT/code_embeddingss.t7")