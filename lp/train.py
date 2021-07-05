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
from sklearn.metrics import roc_auc_score

from model import Model
from preprocess import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor
from arch import archs

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--dataset', type=str, default='Yelp')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + "_h" + str(args.n_hid) + \
         "_drop" + str(args.dropout) + "_epoch" + str(args.epochs) + "_cuda" + str(args.gpu)

logdir = os.path.join("log/eval", args.dataset)
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

    steps_s = [len(meta) for meta in archs[args.dataset]["source"][0]]
    steps_t = [len(meta) for meta in archs[args.dataset]["target"][0]]
    #print(steps_s, steps_t)

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

    model_s = Model(in_dims, args.n_hid, steps_s, dropout = args.dropout).cuda()
    model_t = Model(in_dims, args.n_hid, steps_t, dropout = args.dropout).cuda()

    optimizer = torch.optim.Adam(
        list(model_s.parameters()) + list(model_t.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )

    best_val = None
    final = None
    anchor = None
    for epoch in range(args.epochs):
        train_loss = train(node_feats, node_types, adjs_pt, pos_train, neg_train, model_s, model_t, optimizer)
        val_loss, auc_val, auc_test = infer(node_feats, node_types, adjs_pt, pos_val, neg_val, pos_test, neg_test, model_s, model_t)
        logging.info("Epoch {}; Train err {}; Val err {}; Val auc {}".format(epoch + 1, train_loss, val_loss, auc_val))
        if best_val is None or auc_val > best_val:
            best_val = auc_val
            final = auc_test
            anchor = epoch + 1
    logging.info("Best val auc {} at epoch {}; Test auc {}".format(best_val, anchor, final))

def train(node_feats, node_types, adjs, pos_train, neg_train, model_s, model_t, optimizer):

    model_s.train()
    model_t.train()
    optimizer.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, archs[args.dataset]["source"][0], archs[args.dataset]["source"][1])
    out_t = model_t(node_feats, node_types, adjs, archs[args.dataset]["target"][0], archs[args.dataset]["target"][1])
    loss = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_train[:, 0]], out_t[pos_train[:, 1]]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[neg_train[:, 0]], out_t[neg_train[:, 1]]).sum(dim=-1)))
    loss.backward()
    optimizer.step()
    return loss.item()

def infer(node_feats, node_types, adjs, pos_val, neg_val, pos_test, neg_test, model_s, model_t):

    model_s.eval()
    model_t.eval()
    with torch.no_grad():
        out_s = model_s(node_feats, node_types, adjs, archs[args.dataset]["source"][0], archs[args.dataset]["source"][1])
        out_t = model_t(node_feats, node_types, adjs, archs[args.dataset]["target"][0], archs[args.dataset]["target"][1])
    
    #* validation performance
    pos_val_prod = torch.mul(out_s[pos_val[:, 0]], out_t[pos_val[:, 1]]).sum(dim=-1)
    neg_val_prod = torch.mul(out_s[neg_val[:, 0]], out_t[neg_val[:, 1]]).sum(dim=-1)
    loss = - torch.mean(F.logsigmoid(pos_val_prod) + F.logsigmoid(- neg_val_prod))

    y_true_val = np.zeros((pos_val.shape[0] + neg_val.shape[0]), dtype=np.long)
    y_true_val[:pos_val.shape[0]] = 1
    y_pred_val = np.concatenate((torch.sigmoid(pos_val_prod).cpu().numpy(), torch.sigmoid(neg_val_prod).cpu().numpy()))
    auc_val = roc_auc_score(y_true_val, y_pred_val)

    #* test performance
    pos_test_prod = torch.mul(out_s[pos_test[:, 0]], out_t[pos_test[:, 1]]).sum(dim=-1)
    neg_test_prod = torch.mul(out_s[neg_test[:, 0]], out_t[neg_test[:, 1]]).sum(dim=-1)

    y_true_test = np.zeros((pos_test.shape[0] + neg_test.shape[0]), dtype=np.long)
    y_true_test[:pos_test.shape[0]] = 1
    y_pred_test = np.concatenate((torch.sigmoid(pos_test_prod).cpu().numpy(), torch.sigmoid(neg_test_prod).cpu().numpy()))
    auc_test = roc_auc_score(y_true_test, y_pred_test)
    
    return loss.item(), auc_val, auc_test

if __name__ == '__main__':
    main()