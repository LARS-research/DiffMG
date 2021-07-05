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
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--dataset', type=str, default='DBLP')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50, help='maximum number of training epochs')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=24)
parser.add_argument('--no_norm', action='store_true', default=False, help='disable layer norm')
parser.add_argument('--in_nl', action='store_true', default=False, help='non-linearity after projection')
args = parser.parse_args()

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + "_h" + str(args.n_hid) + \
         "_drop" + str(args.dropout) + "_epoch" + str(args.epochs) + "_cuda" + str(args.gpu)

if args.no_norm is True:
    prefix += "_noLN"
if args.in_nl is True:
    prefix += "_nl"

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
    
    train_idx = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.long).cuda()
    train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.long).cuda()
    valid_idx = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.long).cuda()
    valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.long).cuda()
    test_idx = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.long).cuda()
    test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.long).cuda()

    n_classes = train_target.max().item() + 1
    print("Number of classes: {}".format(n_classes), "Number of node types: {}".format(num_node_types))

    model = Model(node_feats.size(1), args.n_hid, num_node_types, n_classes, steps, dropout = args.dropout, use_norm = not args.no_norm, in_nl = args.in_nl).cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    best_val = None
    final = None
    anchor = None
    patience = 0
    for epoch in range(args.epochs):
        train_loss = train(node_feats, node_types, adjs_pt, train_idx, train_target, model, optimizer)
        val_loss, f1_val, f1_test = infer(node_feats, node_types, adjs_pt, valid_idx, valid_target, test_idx, test_target, model)
        logging.info("Epoch {}; Train err {}; Val err {}".format(epoch + 1, train_loss, val_loss))
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            final = f1_test
            anchor = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience == 10:
                break
    logging.info("Best val {} at epoch {}; Test score {}".format(best_val, anchor, final))

def train(node_feats, node_types, adjs, train_idx, train_target, model, optimizer):

    model.train()
    optimizer.zero_grad()
    out = model(node_feats, node_types, adjs, archs[args.dataset][0], archs[args.dataset][1])
    loss = F.cross_entropy(out[train_idx], train_target)
    loss.backward()
    optimizer.step()
    return loss.item()

def infer(node_feats, node_types, adjs, valid_idx, valid_target, test_idx, test_target, model):

    model.eval()
    with torch.no_grad():
        out = model(node_feats, node_types, adjs, archs[args.dataset][0], archs[args.dataset][1])
    loss = F.cross_entropy(out[valid_idx], valid_target)
    f1_val = f1_score(valid_target.cpu().numpy(), torch.argmax(out[valid_idx], dim=-1).cpu().numpy(), average='macro')
    f1_test = f1_score(test_target.cpu().numpy(), torch.argmax(out[test_idx], dim=-1).cpu().numpy(), average='macro')
    return loss.item(), f1_val, f1_test

if __name__ == '__main__':
    main()