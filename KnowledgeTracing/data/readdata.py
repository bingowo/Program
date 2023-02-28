import numpy as np
from data.DKTDataSet import DKTDataSet
import itertools
import tqdm
import pandas as pd

def create_word_index_table(vocab):
    """
    Creating word to index table
    Input:
    vocab: list. The list of the node vocabulary

    """
    ixtoword = {}
    # period at the end of the sentence. make first dimension be end token
    ixtoword[0] = 'END'
    ixtoword[1] = 'UNK'
    wordtoix = {}
    wordtoix['END'] = 0
    wordtoix['UNK'] = 1
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    return wordtoix, ixtoword

def convert_to_idx(sample, node_word_index, path_word_index):
    """
    Converting to the index 
    Input:
    sample: list. One single training sample, which is a code, represented as a list of neighborhoods.
    node_word_index: dict. The node to word index dictionary.
    path_word_index: dict. The path to word index dictionary.

    """
    sample_index = []
    for line in sample:
        components = line.split(",")
        if len(components) == 3:
            if components[0] in node_word_index:
                starting_node = node_word_index[components[0]]
            else:
                starting_node = node_word_index['UNK']
            if components[1] in path_word_index:
                path = path_word_index[components[1]]
            else:
                path = path_word_index['UNK']
            if components[2] in node_word_index:
                ending_node = node_word_index[components[2]]
            else:
                ending_node = node_word_index['UNK']
            
            sample_index.append([starting_node,path,ending_node])
        else:
            sample_index.append([0,0,0])
    return sample_index

MAX_CODE_LEN = 100


class DataReader():
    def __init__(self, path, maxstep, numofques):
        self.path = path
        self.maxstep = maxstep
        self.numofques = numofques

    def path_prework(self):
        data = []
        code_df = pd.read_csv("EOJDataset/contest513/labeled_paths.tsv",sep="\t")
        # all_training_code = code_df['Unnamed: 0']  #first columns
        all_training_code = code_df['0']
        separated_code = []
        for code in all_training_code:
            if type(code) == str:
                separated_code.append(code.split("@"))
        
        node_hist = {}
        path_hist = {}
        for paths in separated_code:
            tmp = []
            for p in paths:
                if len(p.split(",")) == 3:
                    tmp.append(p)
            paths = tmp

            starting_nodes = [p.split(",")[0] for p in paths]
            path = [p.split(",")[1] for p in paths]
            ending_nodes = [p.split(",")[2] for p in paths]
            nodes = starting_nodes + ending_nodes
            for n in nodes:
                if not n in node_hist:
                    node_hist[n] = 1
                else:
                    node_hist[n] += 1
            for p in path:
                if not p in path_hist:
                    path_hist[p] = 1
                else:
                    path_hist[p] += 1

        node_count = len(node_hist)
        path_count = len(path_hist)
        # np.save("np_counts.npy", [node_count, path_count])
        print("node_count:", node_count)
        print("path_count:", path_count)

        # small frequency then abandon, for node and path
        valid_node = [node for node, count in node_hist.items()]
        valid_path = [path for path, count in path_hist.items()]

        # create ixtoword and wordtoix lists
        node_word_index, node_index_word = create_word_index_table(valid_node)
        path_word_index, path_index_word = create_word_index_table(valid_path)

        tmp = dict()
        for i in range(len(code_df)):
            tmp2 = code_df.loc[i,'Unnamed: 0']
            tmp3 = code_df.loc[i,'0']
            tmp[tmp2] = tmp3

        return tmp, node_word_index, path_word_index

    def getData(self):
        code_df, node_word_index, path_word_index = self.path_prework()

        trainqus = np.array([])
        trainans = np.array([])
        trainpath = np.array([])
        with open(self.path, 'r') as train:
            for lens, ques, ans, paths_id in tqdm.tqdm(itertools.zip_longest(*[train] * 4), desc='loading data:    ', mininterval=2):
                lens = int(lens.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(np.int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(np.int)
                path_id = np.array(paths_id.strip().strip(',').split(',')).astype(np.int)

                path = []
                for p_id in path_id:
                    tmp = code_df[p_id].split("@")
                    raw_features = convert_to_idx(tmp, node_word_index, path_word_index)
                    if len(raw_features) < MAX_CODE_LEN:
                            raw_features += [[0,0,0]]*(MAX_CODE_LEN - len(raw_features))
                    else:
                        raw_features = raw_features[:MAX_CODE_LEN]

                    features = np.array(raw_features).reshape(-1, MAX_CODE_LEN*3)
                    path.append(features)
                
                mod = 0 if lens%self.maxstep == 0 else (self.maxstep - lens%self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                path_zero = np.append(path, np.zeros(mod*3*MAX_CODE_LEN))
                trainqus = np.append(trainqus, ques).astype(np.int)
                trainans = np.append(trainans, ans).astype(np.int)
                trainpath = np.append(trainpath, path_zero).astype(np.int)

        return trainqus.reshape([-1, self.maxstep]), trainans.reshape([-1, self.maxstep]), trainpath.reshape([-1, self.maxstep, 3*MAX_CODE_LEN])