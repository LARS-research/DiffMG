import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import os
import sys
import pickle

##! cstr includes identity but not zero
cstr_source = {  #* u-side 
    "Amazon" : [0, 8],
    "Douban_Movie" : [0, 1, 3, 11],
    "Yelp" : [0, 1, 3, 9]
}

cstr_target = {  #* i-side
    "Amazon" : [1, 2, 4, 6, 8],
    "Douban_Movie" : [2, 5, 7, 9, 11],
    "Yelp" : [2, 5, 7, 9]
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

def preprocess_amazon(prefix):
    #* indices start from 0
    ui = pd.read_csv(os.path.join(prefix, "user_item.dat"), encoding='utf-8', delimiter='\t', names=['uid', 'iid', 'rating', 'time']).drop_duplicates(subset=['uid', 'iid']).reset_index(drop=True)
    ib = pd.read_csv(os.path.join(prefix, "item_brand.dat"), encoding='utf-8', delimiter=',', names=['iid', 'bid']).drop_duplicates().reset_index(drop=True) ## item full
    ic = pd.read_csv(os.path.join(prefix, "item_category.dat"), encoding='utf-8', delimiter=',', names=['iid', 'cid']).drop_duplicates().reset_index(drop=True) ## full
    iv = pd.read_csv(os.path.join(prefix, "item_view.dat"), encoding='utf-8', delimiter=',', names=['iid', 'vid']).drop_duplicates().reset_index(drop=True) ## item not full
    u_num = ui['uid'].unique().shape[0]
    i_num = ui['iid'].unique().shape[0]
    print(u_num, i_num)
    
    #! unconnected pairs
    ui = ui.sort_values(by=['uid', 'iid'], ascending=[True, True]).reset_index(drop=True)
    if not os.path.exists("./preprocessed/Amazon/unconnected_pairs_offset.npy"):
        unconnected_pairs_offset = []
        count = 0
        for u in range(u_num):
            for i in range(i_num):
                if count < ui.shape[0]:
                    if i == ui.iloc[count]['iid'] and u == ui.iloc[count]['uid']:
                        count += 1
                    else:
                        unconnected_pairs_offset.append([u, i + u_num])
                else:
                    unconnected_pairs_offset.append([u, i + u_num])
        assert(count == ui.shape[0])
        assert(count + len(unconnected_pairs_offset) == u_num * i_num)
        np.save("./preprocessed/Amazon/unconnected_pairs_offset", np.array(unconnected_pairs_offset))
    
    offsets = {'i' : u_num, 'b' : u_num + i_num}
    offsets['c'] = offsets['b'] + ib['bid'].max() + 1
    offsets['v'] = offsets['c'] + ic['cid'].max() + 1

    #* node types
    node_types = np.zeros((offsets['v'] + iv['vid'].max() + 1,), dtype=np.int32)
    node_types[offsets['i']:offsets['b']] = 1
    node_types[offsets['b']:offsets['c']] = 2
    node_types[offsets['c']:offsets['v']] = 3
    node_types[offsets['v']:] = 4
    if not os.path.exists("./preprocessed/Amazon/node_types.npy"):
        np.save("./preprocessed/Amazon/node_types", node_types)
    
    #* positive pairs
    ui_pos = ui[ui['rating'] > 3].to_numpy()[:, :2]

    #! negative rating
    neg_ratings = ui[ui['rating'] < 4].to_numpy()[:, :2]
    assert(ui_pos.shape[0] + neg_ratings.shape[0] == ui.shape[0])
    neg_ratings[:, 1] += offsets['i']
    np.save("./preprocessed/Amazon/neg_ratings_offset", neg_ratings)

    indices = np.arange(ui_pos.shape[0])
    np.random.shuffle(indices)
    keep, mask = np.array_split(indices, 2)
    np.random.shuffle(mask)
    train, val, test = np.array_split(mask, [int(len(mask) * 0.6), int(len(mask) * 0.8)])
    
    ui_pos_train = ui_pos[train]
    ui_pos_val = ui_pos[val]
    ui_pos_test = ui_pos[test]
    
    ui_pos_train[:, 1] += offsets['i']
    ui_pos_val[:, 1] += offsets['i']
    ui_pos_test[:, 1] += offsets['i']
    np.savez("./preprocessed/Amazon/pos_pairs_offset", train=ui_pos_train, val=ui_pos_val, test=ui_pos_test)    

    #* adjs with offset
    adjs_offset = {}
    
    ## ui
    ui_pos_keep = ui_pos[keep]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[ui_pos_keep[:, 0], ui_pos_keep[:, 1] + offsets['i']] = 1
    adjs_offset['1'] = sp.coo_matrix(adj_offset)

    ## ib
    ib_npy = ib.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[ib_npy[:, 0] + offsets['i'], ib_npy[:, 1] + offsets['b']] = 1
    adjs_offset['2'] = sp.coo_matrix(adj_offset)

    ## ic
    ic_npy = ic.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[ic_npy[:, 0] + offsets['i'], ic_npy[:, 1] + offsets['c']] = 1
    adjs_offset['3'] = sp.coo_matrix(adj_offset)

    ## iv
    iv_npy = iv.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[iv_npy[:, 0] + offsets['i'], iv_npy[:, 1] + offsets['v']] = 1
    adjs_offset['4'] = sp.coo_matrix(adj_offset)

    f2 = open("./preprocessed/Amazon/adjs_offset.pkl", "wb")
    pickle.dump(adjs_offset, f2)
    f2.close()

def preprocess_douban_movie(prefix):
    um = pd.read_csv(os.path.join(prefix, "user_movie.dat"), encoding='utf-8', delimiter='\t', names=['uid', 'mid', 'rating']).drop_duplicates(subset=['uid', 'mid']).reset_index(drop=True)
    uu = pd.read_csv(os.path.join(prefix, "user_user.dat"), encoding='utf-8', delimiter='\t', names=['u1', 'u2', 'weight']).drop_duplicates(subset=['u1', 'u2']).reset_index(drop=True) ## not full; asym
    ug = pd.read_csv(os.path.join(prefix, "user_group.dat"), encoding='utf-8', delimiter='\t', names=['uid', 'gid', 'weight']).drop_duplicates(subset=['uid', 'gid']).reset_index(drop=True) ## user not full
    ma = pd.read_csv(os.path.join(prefix, "movie_actor.dat"), encoding='utf-8', delimiter='\t', names=['mid', 'aid', 'weight']).drop_duplicates(subset=['mid', 'aid']).reset_index(drop=True) ## 33572; movie not full
    md = pd.read_csv(os.path.join(prefix, "movie_director.dat"), encoding='utf-8', delimiter='\t', names=['mid', 'did', 'weight']).drop_duplicates(subset=['mid', 'did']).reset_index(drop=True) ## movie not full
    mt = pd.read_csv(os.path.join(prefix, "movie_type.dat"), encoding='utf-8', delimiter='\t', names=['mid', 'tid', 'weight']).drop_duplicates(subset=['mid', 'tid']).reset_index(drop=True) ## movie not full
    u_num = um['uid'].unique().shape[0]
    m_num = um['mid'].unique().shape[0]
    print(u_num, m_num)

    #! unconnected pairs
    um = um.sort_values(by=['uid', 'mid'], ascending=[True, True]).reset_index(drop=True)
    if not os.path.exists("./preprocessed/Douban_Movie/unconnected_pairs_offset.npy"):
        unconnected_pairs_offset = []
        count = 0
        for u in range(u_num):
            for m in range(m_num):
                if count < um.shape[0]:
                    if m + 1 == um.iloc[count]['mid'] and u + 1 == um.iloc[count]['uid']:
                        count += 1
                    else:
                        unconnected_pairs_offset.append([u, m + u_num])
                else:
                    unconnected_pairs_offset.append([u, m + u_num])
        assert(count == um.shape[0])
        assert(count + len(unconnected_pairs_offset) == u_num * m_num)
        np.save("./preprocessed/Douban_Movie/unconnected_pairs_offset", np.array(unconnected_pairs_offset))

    offsets = {'m' : u_num, 'g' : u_num + m_num}
    offsets['a'] = offsets['g'] + ug['gid'].max()
    offsets['d'] = offsets['a'] + ma['aid'].max()
    offsets['t'] = offsets['d'] + md['did'].max()

    #* node types 
    node_types = np.zeros((offsets['t'] + mt['tid'].max(),), dtype=np.int32)
    node_types[offsets['m']:offsets['g']] = 1
    node_types[offsets['g']:offsets['a']] = 2
    node_types[offsets['a']:offsets['d']] = 3
    node_types[offsets['d']:offsets['t']] = 4
    node_types[offsets['t']:] = 5
    if not os.path.exists("./preprocessed/Douban_Movie/node_types.npy"):
        np.save("./preprocessed/Douban_Movie/node_types", node_types)
    
    #* positive pairs
    um_pos = um[um['rating'] > 3].to_numpy()[:, :2] - 1

    #! negative rating
    neg_ratings = um[um['rating'] < 4].to_numpy()[:, :2] - 1
    assert(um_pos.shape[0] + neg_ratings.shape[0] == um.shape[0])
    neg_ratings[:, 1] += offsets['m']
    np.save("./preprocessed/Douban_Movie/neg_ratings_offset", neg_ratings)

    indices = np.arange(um_pos.shape[0])
    np.random.shuffle(indices)
    keep, mask = np.array_split(indices, 2)
    np.random.shuffle(mask)
    train, val, test = np.array_split(mask, [int(len(mask) * 0.6), int(len(mask) * 0.8)])
    um_pos_train = um_pos[train]
    um_pos_val = um_pos[val]
    um_pos_test = um_pos[test]
    
    um_pos_train[:, 1] += offsets['m']
    um_pos_val[:, 1] += offsets['m']
    um_pos_test[:, 1] += offsets['m']
    np.savez("./preprocessed/Douban_Movie/pos_pairs_offset", train=um_pos_train, val=um_pos_val, test=um_pos_test)

    #* adjs with offset
    adjs_offset = {}
    
    ## um
    um_pos_keep = um_pos[keep]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[um_pos_keep[:, 0], um_pos_keep[:, 1] + offsets['m']] = 1
    adjs_offset['1'] = sp.coo_matrix(adj_offset)

    ## uu
    uu_swap = pd.DataFrame({'u1' : uu['u2'], 'u2' : uu['u1'], 'weight' : uu['weight']})
    uu_sym = pd.concat([uu, uu_swap]).drop_duplicates().reset_index(drop=True)
    uu_npy = uu_sym.to_numpy()[:, :2] - 1

    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[uu_npy[:, 0], uu_npy[:, 1]] = 1
    adjs_offset['0'] = sp.coo_matrix(adj_offset)

    ## ug
    ug_npy = ug.to_numpy()[:, :2] - 1
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[ug_npy[:, 0], ug_npy[:, 1] + offsets['g']] = 1
    adjs_offset['2'] = sp.coo_matrix(adj_offset)

    ## ma
    ma_npy = ma.to_numpy()[:, :2] - 1
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[ma_npy[:, 0] + offsets['m'], ma_npy[:, 1] + offsets['a']] = 1
    adjs_offset['3'] = sp.coo_matrix(adj_offset)

    ## md
    md_npy = md.to_numpy()[:, :2] - 1
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[md_npy[:, 0] + offsets['m'], md_npy[:, 1] + offsets['d']] = 1
    adjs_offset['4'] = sp.coo_matrix(adj_offset)

    ## mt
    mt_npy = mt.to_numpy()[:, :2] - 1
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[mt_npy[:, 0] + offsets['m'], mt_npy[:, 1] + offsets['t']] = 1
    adjs_offset['5'] = sp.coo_matrix(adj_offset)

    f2 = open("./preprocessed/Douban_Movie/adjs_offset.pkl", "wb")
    pickle.dump(adjs_offset, f2)
    f2.close()


def preprocess_yelp(prefix):
    ub = pd.read_csv(os.path.join(prefix, "user_business.dat"), encoding='utf-8', delimiter='\t', names=['uid', 'bid', 'rating'])
    uu = pd.read_csv(os.path.join(prefix, "user_user.dat"), encoding='utf-8', delimiter='\t', names=['u1', 'u2', 'weight']).drop_duplicates().reset_index(drop=True) ## not full; sym
    uco = pd.read_csv(os.path.join(prefix, "user_compliment.dat"), encoding='utf-8', delimiter='\t', names=['uid', 'coid', 'weight']) ## user not full
    bca = pd.read_csv(os.path.join(prefix, "business_category.dat"), encoding='utf-8', delimiter='\t', names=['bid', 'caid', 'weight']) ## business not full
    bc = pd.read_csv(os.path.join(prefix, "business_city.dat"), encoding='utf-8', delimiter='\t', names=['bid', 'cid', 'weight']) ## business not full
    u_num = ub['uid'].unique().shape[0]
    b_num = ub['bid'].unique().shape[0]
    print(u_num, b_num)
    
    #! unconnected pairs
    if not os.path.exists("./preprocessed/Yelp/unconnected_pairs_offset.npy"):
        unconnected_pairs_offset = []
        count = 0
        for u in range(u_num):
            for b in range(b_num):
                if count < ub.shape[0]:
                    if b + 1 == ub.iloc[count]['bid'] and u + 1 == ub.iloc[count]['uid']:
                        count += 1
                    else:
                        unconnected_pairs_offset.append([u, b + u_num])
                else:
                    unconnected_pairs_offset.append([u, b + u_num])
        assert(count == ub.shape[0])
        assert(count + len(unconnected_pairs_offset) == u_num * b_num)
        np.save("./preprocessed/Yelp/unconnected_pairs_offset", np.array(unconnected_pairs_offset))
    
    offsets = {'b' : u_num, 'co' : u_num + b_num}
    offsets['ca'] = offsets['co'] + uco['coid'].max()
    offsets['c'] = offsets['ca'] + bca['caid'].max()

    #* node types 
    node_types = np.zeros((offsets['c'] + bc['cid'].max(),), dtype=np.int32)
    node_types[offsets['b']:offsets['co']] = 1
    node_types[offsets['co']:offsets['ca']] = 2
    node_types[offsets['ca']:offsets['c']] = 3
    node_types[offsets['c']:] = 4
    if not os.path.exists("./preprocessed/Yelp/node_types.npy"):
        np.save("./preprocessed/Yelp/node_types", node_types)
    
    #* positive pairs
    ub_pos = ub[ub['rating'] > 3].to_numpy()[:, :2] - 1

    #! negative rating
    neg_ratings = ub[ub['rating'] < 4].to_numpy()[:, :2] - 1
    assert(ub_pos.shape[0] + neg_ratings.shape[0] == ub.shape[0])
    neg_ratings[:, 1] += offsets['b']
    np.save("./preprocessed/Yelp/neg_ratings_offset", neg_ratings)

    indices = np.arange(ub_pos.shape[0])
    np.random.shuffle(indices)
    keep, mask = np.array_split(indices, 2)
    np.random.shuffle(mask)
    train, val, test = np.array_split(mask, [int(len(mask) * 0.6), int(len(mask) * 0.8)])
    
    ub_pos_train = ub_pos[train]
    ub_pos_val = ub_pos[val]
    ub_pos_test = ub_pos[test]
    
    ub_pos_train[:, 1] += offsets['b']
    ub_pos_val[:, 1] += offsets['b']
    ub_pos_test[:, 1] += offsets['b']
    np.savez("./preprocessed/Yelp/pos_pairs_offset", train=ub_pos_train, val=ub_pos_val, test=ub_pos_test)

    #* adjs with offset
    adjs_offset = {}
    
    ## ub
    ub_pos_keep = ub_pos[keep]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[ub_pos_keep[:, 0], ub_pos_keep[:, 1] + offsets['b']] = 1
    adjs_offset['1'] = sp.coo_matrix(adj_offset)

    ## uu
    uu_swap = pd.DataFrame({'u1' : uu['u2'], 'u2' : uu['u1'], 'weight' : uu['weight']})
    uu_sym = pd.concat([uu, uu_swap]).drop_duplicates().reset_index(drop=True)
    uu_npy = uu_sym.to_numpy()[:, :2] - 1

    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[uu_npy[:, 0], uu_npy[:, 1]] = 1
    adjs_offset['0'] = sp.coo_matrix(adj_offset)

    ## uco
    uco_npy = uco.to_numpy()[:, :2] - 1
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[uco_npy[:, 0], uco_npy[:, 1] + offsets['co']] = 1
    adjs_offset['2'] = sp.coo_matrix(adj_offset)

    ## bca
    bca_npy = bca.to_numpy()[:, :2] - 1
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[bca_npy[:, 0] + offsets['b'], bca_npy[:, 1] + offsets['ca']] = 1
    adjs_offset['3'] = sp.coo_matrix(adj_offset)

    ## bc
    bc_npy = bc.to_numpy()[:, :2] - 1
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[bc_npy[:, 0] + offsets['b'], bc_npy[:, 1] + offsets['c']] = 1
    adjs_offset['4'] = sp.coo_matrix(adj_offset)

    f2 = open("./preprocessed/Yelp/adjs_offset.pkl", "wb")
    pickle.dump(adjs_offset, f2)
    f2.close()
    

if __name__ == '__main__':
    prefix = "./data/"
    dataset = sys.argv[1]
    prefix = os.path.join(prefix, dataset)
    np.random.seed(int(sys.argv[2]))

    if dataset == "Yelp":
        preprocess_yelp(prefix)
    elif dataset == "Douban_Movie":
        preprocess_douban_movie(prefix)
    elif dataset == "Amazon":
        preprocess_amazon(prefix)
    
    ##! Yelp 1
    ##! Amazon 3
    ##! Douban_Movie 5
