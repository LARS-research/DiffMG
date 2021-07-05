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
import time

from model_search import Model
from preprocess import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor 
from preprocess import cstr_source, cstr_target

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--alr', type=float, default=3e-4, help='learning rate for architecture parameters')
parser.add_argument('--steps_s', type=int, nargs='+', help='number of intermediate states in the meta graph for source node type')
parser.add_argument('--steps_t', type=int, nargs='+', help='number of intermediate states in the meta graph for target node type')
parser.add_argument('--dataset', type=str, default='Yelp')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for supernet training')
parser.add_argument('--eps', type=float, default=0., help='probability of random sampling')
parser.add_argument('--decay', type=float, default=0.9, help='decay factor for eps')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + \
         "_h" + str(args.n_hid) + "_alr" + str(args.alr) + \
         "_s" + str(args.steps_s) + "_t" + str(args.steps_t) + "_epoch" + str(args.epochs) + \
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

    datadir = "preprocessed"
    prefix = os.path.join(datadir, args.dataset)

    #* load data
    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).cuda()

    adjs_offset = pickle.load(open(os.path.join(prefix, "adjs_offset.pkl"), "rb"))
    adjs_pt = []
    if '0' in adjs_offset:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_sym(adjs_offset['0'] + sp.eye(adjs_offset['0'].shape[0], dtype=np.float32))).cuda())
    for i in range(1, int(max(adjs_offset.keys())) + 1):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)].T + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
    adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_offset['1'].shape[0], dtype=np.float32).tocoo()).cuda())
    adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset['1'].shape).cuda())
    print("Loading {} adjs...".format(len(adjs_pt)))

    #* load labels
    pos = np.load(os.path.join(prefix, "pos_pairs_offset.npz"))
    pos_train = pos['train']
    pos_val = pos['val']
    pos_test = pos['test']

    neg = np.load(os.path.join(prefix, "neg_pairs_offset.npz"))
    neg_train = neg['train']
    neg_val = neg['val']
    neg_test = neg['test']

    #* one-hot IDs as input features
    in_dims = []
    node_feats = []
    for k in range(num_node_types):
        in_dims.append((node_types == k).sum().item())
        i = torch.stack((torch.arange(in_dims[-1], dtype=torch.long), torch.arange(in_dims[-1], dtype=torch.long)))
        v = torch.ones(in_dims[-1])
        node_feats.append(torch.sparse.FloatTensor(i, v, torch.Size([in_dims[-1], in_dims[-1]])).cuda())
    assert(len(in_dims) == len(node_feats))   

    model_s = Model(in_dims, args.n_hid, len(adjs_pt), args.steps_s, cstr_source[args.dataset]).cuda()
    model_t = Model(in_dims, args.n_hid, len(adjs_pt), args.steps_t, cstr_target[args.dataset]).cuda()

    optimizer_w = torch.optim.Adam(
        list(model_s.parameters()) + list(model_t.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )

    optimizer_a = torch.optim.Adam(
        model_s.alphas() + model_t.alphas(),
        lr=args.alr
    )

    eps = args.eps
    start_t = time.time()
    for epoch in range(args.epochs):
        train_error, val_error = train(node_feats, node_types, adjs_pt, pos_train, neg_train, pos_val, neg_val, model_s, model_t, optimizer_w, optimizer_a, eps)
        logging.info("Epoch {}; Train err {}; Val err {}; Source arch {}; Target arch {}".format(epoch + 1, train_error, val_error, model_s.parse(), model_t.parse()))
        eps = eps * args.decay
    end_t = time.time()
    print("Search time (in minutes): {}".format((end_t - start_t) / 60))

def train(node_feats, node_types, adjs, pos_train, neg_train, pos_val, neg_val, model_s, model_t, optimizer_w, optimizer_a, eps):

    idxes_seq_s, idxes_res_s = model_s.sample(eps)
    idxes_seq_t, idxes_res_t = model_t.sample(eps)

    optimizer_w.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, idxes_seq_s, idxes_res_s)
    out_t = model_t(node_feats, node_types, adjs, idxes_seq_t, idxes_res_t)
    loss_w = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_train[:, 0]], out_t[pos_train[:, 1]]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[neg_train[:, 0]], out_t[neg_train[:, 1]]).sum(dim=-1)))
    loss_w.backward()
    optimizer_w.step()

    optimizer_a.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, idxes_seq_s, idxes_res_s)
    out_t = model_t(node_feats, node_types, adjs, idxes_seq_t, idxes_res_t)
    loss_a = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_val[:, 0]], out_t[pos_val[:, 1]]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[neg_val[:, 0]], out_t[neg_val[:, 1]]).sum(dim=-1)))
    loss_a.backward()
    optimizer_a.step()

    return loss_w.item(), loss_a.item()

if __name__ == '__main__':
    main()