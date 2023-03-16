import json
import tree_tools
import word2vec
import torch
from tqdm import tqdm

with open("EOJDataset/AST/ast_trees.json",'r') as f:
    trees = json.load(f)

words = []
for number, tree in tqdm(trees.items()):
    tree = tree_tools.tree_VE_prune(tree, 512)
    trees[number] = tree
    words.extend(tree[0])
    # print(type(tree),len(tree[0]),len(tree[1]))
with open("EOJDataset/AST/ast_trees_pruned.json",'w') as f:
    f.write(json.dumps(trees))

print("total word:",len(words))
cache_word2vec = word2vec.create_word_dict(words)
print("unique word:",len(cache_word2vec))

def save_node_masks(trees):
    for number, tree in tqdm(trees.items()):
        tree = tree_tools.tree_VE_to_tensor(tree, cache_word2vec)
        trees[number] = tree
        # print(type(tree),len(tree[0]),len(tree[1]))
    torch.save(trees,"EOJDataset/AST/node_masks.t7")

def print_code_lens(trees):
    lens = 512
    cnt = [0] * lens
    for number, tree in trees.items():
        node,_ = tree
        cnt[min(lens-1,len(node))] += 1
    for i in range(lens):
        if cnt[i] != 0:
            print(i,cnt[i])

