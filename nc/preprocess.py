import os
import sys
import numpy as np
import torch
import pickle as pkl
import scipy.sparse as sp

cstr_nc = {
    "DBLP" : [1, 4],
    "ACM" : [0, 2, 4],
    "IMDB" : [0, 2, 4]
}

def normalize_sym(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_row(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def main(dataset):
    prefix = os.path.join("./data/", dataset)
    with open(os.path.join(prefix, "edges.pkl"), "rb") as f:
        edges = pkl.load(f)
        f.close()

    node_types = np.zeros((edges[0].shape[0],), dtype=np.int32)
    
    a = np.unique(list(edges[0].tocoo().row) + list(edges[2].tocoo().row))
    b = np.unique(edges[0].tocoo().col)
    c = np.unique(edges[2].tocoo().col)
    print(a.shape[0], b.shape[0], c.shape[0])
    assert(a.shape[0] + b.shape[0] + c.shape[0] == node_types.shape[0])
    assert(np.unique(np.concatenate((a, b, c))).shape[0] == node_types.shape[0])

    node_types[a.shape[0]:a.shape[0] + b.shape[0]] = 1
    node_types[a.shape[0] + b.shape[0]:] = 2
    assert(node_types.sum() == b.shape[0] + 2 * c.shape[0])
    np.save(os.path.join(prefix, "node_types"), node_types)
    
if __name__ == "__main__":
    main("DBLP")
    main("ACM")
    main("IMDB")