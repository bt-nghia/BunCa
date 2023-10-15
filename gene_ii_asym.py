import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from tqdm import tqdm


def get_graph(path, x, y):
    with open(os.path.join(path), 'r') as f:
        b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

    indice = np.array(b_i_pairs, dtype=np.int32)
    values = np.ones(len(b_i_pairs), dtype=np.float32)
    b_i_graph = sp.coo_matrix(
        (values, (indice[:, 0], indice[:, 1])), shape=(x, y)).tocsr()
    return b_i_graph


def save_sp_mat(csr_mat, name):
    sp.save_npz(name, csr_mat)


def load_sp_mat(name):
    return sp.load_npz(name)

def filter(threshold, mat):
    mask = mat >= threshold
    mat = mat * mask
    return mat

def gen_ii_asym(ix_mat, threshold=0):
    '''
    mat: ui or bi
    '''
    ii_co = ix_mat @ ix_mat.T
    i_count = ix_mat.sum(axis=1)
    i_count += (i_count == 0) # mask all zero with 1
    # norm_ii = normalize(ii_asym, norm='l1', axis=1)
    # return norm_ii
    # return ii_asym
    mask = ii_co > threshold
    ii_co = ii_co.multiply(mask)
    ii_asym = ii_co / i_count
    # normalize by row -> asym matrix
    return ii_co


if __name__ == '__main__':
    users, bundles, items = 8039, 4771, 32770
    dir = 'datasets/Youshu'
    path = [dir + '/user_bundle_train.txt',
            dir + '/user_item.txt',
            dir + '/bundle_item.txt']
    
    raw_graph = [get_graph(path[0], users, bundles),
                 get_graph(path[1], users, items),
                 get_graph(path[2], bundles, items)]

    ub, ui, bi = raw_graph

    pbar = tqdm(enumerate([ui.T, bi.T]), total = 2, desc="gene", ncols=100)
    asym_mat = []
    for i, mat in pbar:
        asym_mat.append(gen_ii_asym(mat))

    pbar = tqdm(enumerate(["/iui_asym.npz", "/ibi_asym.npz"]), total = 2, desc="save", ncols=100)
    for i, data in pbar:
        save_sp_mat(asym_mat[i], dir + data)