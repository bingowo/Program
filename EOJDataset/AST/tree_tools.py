from typing import *

import torch
import math
import numpy

TreeV = List[str]
TreeE = Tuple[List[int], List[int]]
TreeVE = Tuple[TreeV, TreeE]
TreeTensor = Tuple[torch.Tensor, torch.Tensor]


def tree_VE_prune(tree_VE: TreeVE, max_node_count = 512) -> TreeVE:
    tree_V, Tree_E = tree_VE
    n = len(tree_V)
    if n <= max_node_count:
        return tree_VE

    v_out, v_in = Tree_E
    count = [0] * len(tree_V)
    for v in v_in:
        count[v] += 1
    
    can_prune = [i for i in range(n) if count[i] == 0]
    assert(len(can_prune))
    numpy.random.shuffle(can_prune)

    k = min(len(can_prune), n - max_node_count)
    pruned = set(can_prune[: k])

    pruned_V = []
    vid_map = list(range(n))
    for idx, v in enumerate(tree_V):
        if idx not in pruned:
            pruned_V.append(v)
            vid_map[idx] = len(pruned_V) - 1
    
    pruned_out_in = [(out_id, in_id) for out_id, in_id in zip(v_out, v_in) if out_id not in pruned]
    pruned_out = [vid_map[out_id] for out_id, _ in pruned_out_in]
    pruned_in = [vid_map[in_id] for _, in_id in pruned_out_in]

    assert(len(pruned_V) - 1 == len(pruned_in))
    assert(len(pruned_V) < len(tree_V))

    return tree_VE_prune((pruned_V, (pruned_out, pruned_in)), max_node_count)


def tree_VE_to_tensor(tree_VE: TreeVE, word2vec_cache: Dict[str, torch.Tensor] = None) -> TreeTensor:
    def word2vec(word: str):
        if word2vec_cache and word in word2vec_cache:
            return word2vec_cache[word]
        import word2vec
        return word2vec.word2vec(word)
    
    tree_V, tree_E = tree_VE
    
    nodes = torch.stack([word2vec(v) for v in tree_V])
    edges = torch.tensor(tree_E, dtype=torch.long)

    n, _ = nodes.shape
    _, m = edges.shape
    mask = torch.eye(n, dtype=torch.bool)

    for i in range(m):
        mask[edges[1, i]] = torch.logical_or(mask[edges[1, i]], mask[edges[0, i]])

    mask = ~mask
    return nodes, mask


def merge_tree_VE(tree_VE1: TreeVE, tree_VE2: TreeVE, merge_node: str) -> TreeVE:
    tree_V1, tree_E1 = tree_VE1
    tree_V2, tree_E2 = tree_VE2

    e1_src, e1_dst = tree_E1
    e2_src, e2_dst = tree_E2
    
    e1_src = [v + 1 for v in e1_src] + [1]
    e1_dst = [v + 1 for v in e1_dst] + [0]
    d = 1 + len(tree_V1)
    e2_src = [v + d for v in e2_src] + [d]
    e2_dst = [v + d for v in e2_dst] + [0]
    
    V = [merge_node] + tree_V1 + tree_V2
    E = (e1_src + e2_src, e1_dst + e2_dst)
    assert(len(V) - 1 == len(E[0]))
    return V, E


def collate_tree_tensor(batch: List[Tuple[int, int, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ques_list = [ques for ques, ans, nodes, mask in batch]
    ans_list = [ans for ques, ans, nodes, mask in batch]
    nodes_list = [nodes for ques, ans, nodes, mask in batch]
    mask_list = [mask for ques, ans, nodes, mask in batch]
    node_batch = torch.nn.utils.rnn.pad_sequence(nodes_list)

    ques_list = torch.tensor(ques_list, dtype=torch.long)
    ans_list = torch.tensor(ans_list, dtype=torch.long)

    n = node_batch.shape[0]

    mask_base = ~torch.eye(n, dtype=torch.bool)
    mask_batch = mask_base.repeat(len(batch), 1, 1)

    for idx, mask in enumerate(mask_list):
        n, m = mask.shape
        mask_batch[idx, :n, :m] = mask

    return ques_list, ans_list, node_batch, mask_batch