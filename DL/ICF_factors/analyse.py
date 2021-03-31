import gzip
import pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pp

import argparse
import os

parser = argparse.ArgumentParser(description='ICF')
parser.add_argument('exps', type=str, default='./exps', 
                    help='Path to experiments folder')
args = parser.parse_args()

import plotting

ls = lambda x : [os.path.join(x,i) for i in os.listdir(x)]

ep_lengths = []

exps = ls(args.exps)

for exp in exps:
    log = pickle.load(gzip.open(os.path.join(exp, 'log.pkl.gz'),'rb'))
    el = log['episode-length']
    print(exp, len(el))
    ep_lengths.append(el[:4000])

for i,exp in zip(ep_lengths, exps):
    c = 'red' if 'a2c' in exp else 'blue'
    plotting.plot_1d_smooth(np.float32(i), N=20, fill=False, pltkw={'alpha':0.5, 'c':c})
pp.savefig('ep_lengths.png')
