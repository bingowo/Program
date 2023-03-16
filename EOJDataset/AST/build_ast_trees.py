import torch
from tqdm import tqdm
import my_parser
import os
import json


def file2vec(filename):
    number = filename.split('\\')[-1].split('_')[1][1:]
    # print(filename)

    with open(filename, encoding='utf-8') as f:
        txt = f.read()

    tree = my_parser.parse(txt)

    return number, tree

if __name__ == '__main__':
    code_dir = 'EOJDataset/contest513/ContestCode'
    code_files = []
    for root, dirs, files in os.walk(code_dir):
        for file in files:
            if 'Compilation' not in file:
                code_files.append(os.path.join(root, file))

    # code_files = ['EOJDataset/AST\\test_#10110_test.c']

    trees = {}
    print(len(code_files))
    cnt = 0
    for file in tqdm(code_files,mininterval=3):
        number, tree = file2vec(file)
        trees[number] = tree

        cnt += 1
        if cnt % 5000 == 0:
            with open("EOJDataset/AST/ast_tree_"+str(cnt)+".json",'w') as f:
                f.write(json.dumps(trees))
            # torch.save(trees,"EOJDataset/AST/ast_tree_"+str(cnt)+".t7")
    with open("EOJDataset/AST/ast_trees.json",'w') as f:    
        f.write(json.dumps(trees))
    # torch.save(trees,"EOJDataset/AST/ast_trees.t7")