import os
import sys
import numpy as np
import torch
import pickle as pkl

cstr_nc = {
    "DBLP" : [1, 4],
    "ACM" : [0, 2, 4],
    "IMDB" : [0, 2, 4]
}

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