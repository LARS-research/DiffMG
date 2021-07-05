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

from model_search import Model
from preprocess import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor 
from preprocess import cstr_nc

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--alr', type=float, default=3e-4, help='learning rate for architecture parameters')
parser.add_argument('--steps', type=int, nargs='+', help='number of intermediate states in the meta graph')
parser.add_argument('--dataset', type=str, default='DBLP')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50, help='number of epochs for supernet training')
parser.add_argument('--eps', type=float, default=0.3, help='probability of random sampling')
parser.add_argument('--decay', type=float, default=0.9, help='decay factor for eps')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + \
         "_h" + str(args.n_hid) + "_alr" + str(args.alr) + \
         "_s" + str(args.steps) + "_epoch" + str(args.epochs) + \
         "_cuda" + str(args.gpu) + "_eps" + str(args.eps) + "_d" + str(args.decay)

logdir = os.path.join("log/search", args.dataset)
if not os.path.exists(logdir):
    os.makedirs(logdir)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(logdir, prefix + ".txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():

    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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
    
    train_idx = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.long).cuda()
    train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.long).cuda()
    valid_idx = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.long).cuda()
    valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.long).cuda()

    n_classes = train_target.max().item() + 1
    print("Number of classes: {}".format(n_classes), "Number of node types: {}".format(num_node_types))

    model = Model(node_feats.size(1), args.n_hid, num_node_types, len(adjs_pt), n_classes, args.steps, cstr_nc[args.dataset]).cuda()

    optimizer_w = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    optimizer_a = torch.optim.Adam(
        model.alphas(),
        lr=args.alr
    )

    eps = args.eps
    for epoch in range(args.epochs):
        train_error, val_error = train(node_feats, node_types, adjs_pt, train_idx, train_target, valid_idx, valid_target, model, optimizer_w, optimizer_a, eps)
        logging.info("Epoch {}; Train err {}; Val err {}; Arch {}".format(epoch + 1, train_error, val_error, model.parse()))
        eps = eps * args.decay

def train(node_feats, node_types, adjs, train_idx, train_target, valid_idx, valid_target, model, optimizer_w, optimizer_a, eps):

    idxes_seq, idxes_res = model.sample(eps)

    optimizer_w.zero_grad()
    out = model(node_feats, node_types, adjs, idxes_seq, idxes_res)
    loss_w = F.cross_entropy(out[train_idx], train_target)
    loss_w.backward()
    optimizer_w.step()

    optimizer_a.zero_grad()
    out = model(node_feats, node_types, adjs, idxes_seq, idxes_res)
    loss_a = F.cross_entropy(out[valid_idx], valid_target)
    loss_a.backward()
    optimizer_a.step()

    return loss_w.item(), loss_a.item()

if __name__ == '__main__':
    main()