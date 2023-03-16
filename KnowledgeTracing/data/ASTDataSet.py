from typing import *
import torch
from torch.utils.data.dataset import Dataset

from ..data.readdata import DataReader
from EOJDataset.AST.tree_tools import tree_VE_to_tensor
from EOJDataset.AST.word2vec import create_word_dict

import json
from tqdm import tqdm

class ASTDataSet(Dataset):
    def __init__(self, data_path: str):
        super().__init__()
        handle = DataReader(data_path, 1)
        ques, ans, code_ids = handle.getData()
        
        # length = 5000
        # ques = ques[:length]
        # ans = ans[:length]
        # code_ids = code_ids[:length]

        self.ques = torch.tensor(ques).long()
        self.ans = torch.tensor(ans).long()
        self.code_ids = code_ids

        with open("EOJDataset/AST/ast_trees_pruned.json",'r') as f:
            self.trees = json.load(f)
        self.cache_word2vec = create_word_dict(trees=self.trees)

        # self.nodes = list()
        # self.mask = list()

        # for index in tqdm(range(len(self.ques))):
        #     nodes, mask = tree_VE_to_tensor(self.trees[str(self.code_ids[index,0])], self.cache_word2vec)
        #     self.nodes.append(nodes)
        #     self.mask.append(mask)
    

    def __len__(self) -> int:
        return len(self.ques)
    
    def __getitem__(self, index) -> Tuple[int, int, torch.tensor, torch.tensor]:
        nodes, mask = tree_VE_to_tensor(self.trees[str(self.code_ids[index,0])], self.cache_word2vec)
        # nodes = self.nodes[index]
        # mask = self.mask[index]
        return self.ques[index,0], self.ans[index,0], nodes, mask
