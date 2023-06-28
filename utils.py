import os
import re
import pickle
import numpy as np
import pandas as pd
import networkx as nx

import torch
from gensim.models.word2vec import Word2Vec, KeyedVectors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def read_pickle(path):
    with open(path, 'rb') as f:
        G = pickle.load(f)
    return G


def get_file_paths(target_dir, extension='.pickle'):
    paths = []
    for root, dirs, files in os.walk(target_dir):
        if len(dirs)==0 and len(files) > 0:
            for f in files:
                if f.endswith(extension):
                    paths.append(os.path.join(root, f))
    return paths


def split_df(df, n_or_frac, column='family', shuffle=True, allow_lower_n=False):
    if type(n_or_frac) is int:
        if allow_lower_n:
            train_df = df.groupby(column).apply(lambda x: x.sample(n=n_or_frac if x.shape[0]>=n_or_frac else x.shape[0])).droplevel(level=0)
        else:
            train_df = df.groupby(column).sample(n=n_or_frac)
                
    else:
        train_df = df.groupby(column).sample(frac=n_or_frac)
    valid_df = df[~df.index.isin(train_df.index)]
    
    if shuffle:
        train_df = train_df.sample(frac=1)
        valid_df = valid_df.sample(frac=1)
    return train_df, valid_df


class AssemblyNormalizer:
    def __init__(self):
        pass
    
    @staticmethod
    def normalize(inst):
        # Follow the paper
        # using x86 instruction set
        # and connect every part with "-"

        # integer       |  N
        # fcn.xxx       |  fcn
        # random string |  M            ## Not done

        asm_normed = []
        inst = inst.replace(" + ", "+")
        inst = inst.replace("-", "")
        inst = re.sub(",", " ", inst).split()

        for _inst in inst:
            _ins = re.split("(\[|\]|\+|\*)", _inst)
            for i in range(len(_ins)):
                if "fcn" in _ins[i]:
                    _ins[i] = "fcn"
                elif "sym" in _ins[i]:
                    _ins[i] = "sym"
                elif "0x" in _ins[i] or _ins[i].isdigit():
                    _ins[i] = "N" 

                # Special strings
                elif "str" in _ins[i]:
                    _ins[i] = "str"
                elif "reloc" in _ins[i]:
                    _ins[i] = "reloc"
                elif "obj" in _ins[i]:
                    _ins[i] = "obj"
                elif "loc" in _ins[i]:
                    _ins[i] = "loc"
            asm_normed.append("".join(_ins))
        return "-".join(asm_normed)

    @staticmethod
    def apply_norm(graph):
        normed_asm = []
        for assembly in nx.get_node_attributes(graph, 'x').values():
            normed_asm.append([AssemblyNormalizer.normalize(asm) for addr, asm in assembly])
        return normed_asm
    
    @staticmethod
    def concat_normed(normed_asm_list):
        return sum(normed_asm_list, [])


class word2vec:
    def __init__(self):
        self.wv = None
    
    def load(self, path):
        self.wv = KeyedVectors.load(path, mmap="r")
        
    def train(self, sentences):
        # sg=0: use CBOW instead of skip-gram
        # cbow_mean: use mean value instead of addition
        # min_count=0: do not remove the words with frequencies 1

        model = Word2Vec(sentences=sentences, vector_size=128, alpha=0.025, 
                         min_count=0, window=5, workers=4, 
                         sg=0, hs=0, cbow_mean=1, epochs=5)
        self.model = model
        self.wv = model.wv

    def infer(self, normed_inst):
        if self.wv is None:
            raise ValueError('Word2Vec model is not loaded yet, load with word2vec.load() first')
        
        if normed_inst in self.wv:
            return self.wv[normed_inst]
        else:
            return np.zeros(self.wv.vector_size, dtype=np.float32)


class DataProcessor(word2vec):
    def __init__(self):
        super().__init__()
        self.normalizer = AssemblyNormalizer()
        
    def from_networkx(self, G):
        normalize = self.normalizer.normalize
        name_dict = {node: i for i, node in enumerate(G.nodes)}

        nodes = torch.zeros((len(G.nodes), self.wv.vector_size))
        for node, i in name_dict.items():
            node_asm = G.nodes[node]['x']
            node_vecs  = np.vstack([self.infer(normalize(asm)) for _, asm in node_asm])
            nodes[i] = torch.tensor(node_vecs.sum(axis=0))

        edges = torch.tensor([[name_dict[edge[0]], name_dict[edge[1]]] for i, edge in enumerate(G.edges)], dtype=torch.long).T
        return nodes, edges


def compute_metrices(y_true, y_pred):
    accuracy  = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall    = recall_score(y_true=y_true, y_pred=y_pred)
    f1        = f1_score(y_true=y_true, y_pred=y_pred)
    metrices  = {'accuracy':  accuracy, 
                 'precision': precision, 
                 'recall':    recall, 
                 'f1':        f1}
    return metrices