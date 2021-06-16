import numpy as np
import scipy.sparse as sp
import os
import torch
import sys

def main(prefix):

    pos_pairs_offset = np.load(os.path.join(prefix, "pos_pairs_offset.npz"))
    unconnected_pairs_offset = np.load(os.path.join(prefix, "unconnected_pairs_offset.npy"))
    neg_ratings_offset = np.load(os.path.join(prefix, "neg_ratings_offset.npy"))

    train_len = pos_pairs_offset['train'].shape[0]
    val_len = pos_pairs_offset['val'].shape[0]
    test_len = pos_pairs_offset['test'].shape[0]
    pos_len = train_len + val_len + test_len

    if pos_len > neg_ratings_offset.shape[0]:
        indices = np.arange(unconnected_pairs_offset.shape[0])
        assert(indices.shape[0] > pos_len)
        np.random.shuffle(indices)
        makeup = indices[:pos_len - neg_ratings_offset.shape[0]]
        neg_ratings_offset = np.concatenate((neg_ratings_offset, unconnected_pairs_offset[makeup]), axis=0)
        assert(pos_len == neg_ratings_offset.shape[0])

    indices = np.arange(neg_ratings_offset.shape[0])
    np.random.shuffle(indices)
    np.savez(os.path.join(prefix, "neg_pairs_offset"), train=neg_ratings_offset[indices[:train_len]],
                                                        val=neg_ratings_offset[indices[train_len:train_len + val_len]],
                                                        test=neg_ratings_offset[indices[train_len + val_len:pos_len]])

if __name__ == '__main__':
    dataset = sys.argv[1]
    prefix = os.path.join("./preprocessed/", dataset)
    np.random.seed(int(sys.argv[2]))
    main(prefix)

    #! Yelp 2
    #! Amazon 4
    #! Douban_Movie 6
