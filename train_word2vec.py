import os
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

from utils import read_pickle, AssemblyNormalizer, word2vec

def writelog(message):
    print('[INFO] [{}] {}'.format(datetime.now(), message))

def parse_args():
    parser = argparse.ArgumentParser(description='Assembly Word2Vec Trainer')
    parser.add_argument('-d', '--graph-dir', type=str, required=True, metavar='<directory>', 
                        help='directory to the FCG pickle files')
    parser.add_argument('-M', '--max-samples', type=int, required=False, metavar='INT', default=None,
                        help='maximum number of FCGs to train Word2Vec (default: None)')
    parser.add_argument('--seed', type=int, required=False, metavar='INT', default=666,
                        help='random seed')
    parser.add_argument('-s', '--save', type=str, required=False, default='model_saved/w2v.model', metavar='<path>', 
                        help='path to save the Word2Vec model')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    np.random.seed(args.seed)

    paths = [os.path.join(args.graph_dir, f) for f in os.listdir(args.graph_dir)]
    np.random.shuffle(paths)
    if (args.max_samples is not None) and (args.max_samples < len(paths)):
        paths = paths[:args.max_samples]
    
    normalizer = AssemblyNormalizer()
    w2v        = word2vec()

    writelog('Load Graphs ...')
    normed = [normalizer.apply_norm(read_pickle(path)) for path in tqdm(paths)]

    writelog('Train Word2Vec model ...')
    sentences = normalizer.concat_normed(normed)
    w2v.train(sentences)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    w2v.model.save(args.save)
    writelog(f'Model saved in {args.save}')

if __name__=='__main__':
    main()
