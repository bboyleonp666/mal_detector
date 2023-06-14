import re
import pickle
import numpy as np
import networkx as nx

import torch
from gensim.models.word2vec import Word2Vec, KeyedVectors


def read_pickle(path):
    with open(path, 'rb') as f:
        G = pickle.load(f)
    return G


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
        self.model = None
    
    def load(self, path):
        self.model = Word2Vec.load(path)
        
    def train(self, sentences):
        # sg=0: use CBOW instead of skip-gram
        # cbow_mean: use mean value instead of addition
        # min_count=0: do not remove the words with frequencies 1

        model = Word2Vec(sentences=sentences, vector_size=128, alpha=0.025, 
                         min_count=0, window=5, workers=4, 
                         sg=0, hs=0, cbow_mean=1, epochs=5)
        self.model = model

    def infer(self, normed_inst):
        if self.model is None:
            raise ValueError('Word2Vec model is not loaded yet, load with word2vec.load() first')
        
        if normed_inst in self.model.wv:
            return self.model.wv[normed_inst]
        else:
            return np.zeros(self.model.wv.vector_size, dtype=np.float32)


class DataProcessor(word2vec):
    def __init__(self):
        super().__init__()
        self.normalizer = AssemblyNormalizer()
        
    def from_networkx(self, G):
        normalize = self.normalizer.normalize
        name_dict = {node: i for i, node in enumerate(G.nodes)}

        nodes = torch.zeros((len(G.nodes), self.model.wv.vector_size))
        for node, i in name_dict.items():
            node_asm = G.nodes[node]['x']
            node_vecs  = np.vstack([self.infer(normalize(asm)) for _, asm in node_asm])
            nodes[i] = torch.tensor(node_vecs.sum(axis=0))

        edges = torch.tensor([[name_dict[edge[0]], name_dict[edge[1]]] for i, edge in enumerate(G.edges)], dtype=torch.long).T

        return nodes, edges