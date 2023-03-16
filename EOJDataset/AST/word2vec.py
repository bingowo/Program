from typing import *

import torch
import functools
import json
from sentence_transformers import SentenceTransformer

@functools.lru_cache()
def word2vec(word: str) -> torch.Tensor:
    sentence2emb = SentenceTransformer('all-MiniLM-L6-v2')
    return sentence2emb.encode(word, device="cuda", show_progress_bar=False, convert_to_tensor=True).cpu()


def create_word_dict(word_list: List[str] = None, path: str = None) -> Dict[str, torch.Tensor]:
    if path != None:
        with open(path,'r') as f:
            trees = json.load(f)
        word_list = []
        for number, tree in trees.items():
            word_list.extend(tree[0])
        
    assert(word_list != None)
    word_list = list(set(word_list))
    
    sentence2emb = SentenceTransformer('all-MiniLM-L6-v2')
    results: List[torch.Tensor] = sentence2emb.encode(word_list, 
                                 show_progress_bar= True, 
                                 batch_size= 128,
                                 convert_to_numpy= False,
                                 device= "cuda")
    results = [result.cpu() for result in results]
    return dict(zip(word_list, results))