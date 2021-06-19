import os
import sys
import time
import numpy as np
import pickle
import scipy.sparse as sp
import logging
import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from model import Model
from preprocess import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor
from arch import archs

parser = argparse.ArgumentParser()
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--dataset', type=str, default='DBLP')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--no_norm', action='store_true', default=False, help='disable layer norm')
parser.add_argument('--in_nl', action='store_true', default=False, help='non-linearity after projection')
args = parser.parse_args()

def main():
    
    torch.cuda.set_device(args.gpu)

    steps = [len(meta) for meta in archs[args.dataset][0]]
    print("Steps: {}".format(steps))

    datadir = "data"
    prefix = os.path.join(datadir, args.dataset)

    #* load data
    with open(os.path.join(prefix, "node_features.pkl"), "rb") as f:
        node_feats = pickle.load(f)
        f.close()
    node_feats = torch.from_numpy(node_feats.astype(np.float32)).cuda()

    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).cuda()

    with open(os.path.join(prefix, "edges.pkl"), "rb") as f:
        edges = pickle.load(f)
        f.close()
    
    adjs_pt = []
    for mx in edges:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(mx.astype(np.float32) + sp.eye(mx.shape[0], dtype=np.float32))).cuda())
    adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(edges[0].shape[0], dtype=np.float32).tocoo()).cuda())
    adjs_pt.append(torch.sparse.FloatTensor(size=edges[0].shape).cuda())
    print("Loading {} adjs...".format(len(adjs_pt)))

    #* load labels
    with open(os.path.join(prefix, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
        f.close()
    
    test_idx = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.long).cuda()
    test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.long).cuda()

    n_classes = test_target.max().item() + 1
    print("Number of classes: {}".format(n_classes), "Number of node types: {}".format(num_node_types))

    model = Model(node_feats.size(1), args.n_hid, num_node_types, n_classes, steps, use_norm = not args.no_norm, in_nl = args.in_nl).cuda()
    model.load_state_dict(torch.load(os.path.join("checkpoint", args.dataset + ".pt"), map_location=torch.device('cuda', args.gpu)))
    f1_test = infer(node_feats, node_types, adjs_pt, test_idx, test_target, model)
    print("Test score: {}".format(f1_test))

def infer(node_feats, node_types, adjs, test_idx, test_target, model):

    model.eval()
    with torch.no_grad():
        out = model(node_feats, node_types, adjs, archs[args.dataset][0], archs[args.dataset][1])
    f1_test = f1_score(test_target.cpu().numpy(), torch.argmax(out[test_idx], dim=-1).cpu().numpy(), average='macro')
    return f1_test

if __name__ == '__main__':
    main()