import argparse
import numpy as np
import scipy.sparse as sp
from gene_ii_co_oc import load_sp_mat
from sklearn.preprocessing import normalize
import torch


def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-d", "--dataset", default="Youshu", type=str, help="dataset to train")
    parser.add_argument("-u", "--userfil", default=4, type=int, help="iui filter")
    parser.add_argument("-b", "--bundlefil", default=4, type=int, help="ibi filter")

    args = parser.parse_args()
    return args

if __name__=='__main__':
    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]
    b_fil = paras["bundlefil"]
    u_fil = paras["userfil"]

    path_iui = f"datasets/{dataset_name}/iui_cooc.npz"
    path_ibi = f"datasets/{dataset_name}/ibi_cooc.npz"
    save_path_iui = f"datasets/{dataset_name}/n_neigh_iui"
    save_path_ibi = f"datasets/{dataset_name}/n_neigh_ibi"

    iui = load_sp_mat(path_iui)
    print(iui.getnnz())
    ibi = load_sp_mat(path_ibi)
    print(ibi.getnnz())

    print("statistic")
    ii_b_max = int(ibi.max())
    print(f"max i-i interactions through b: {ii_b_max}")
    ii_u_max = int(iui.max())
    print(f"max i-i interactions through u: {ii_u_max}")

    # ---------------------- bundle --------------------------
    print("ii bundle")
    for i in range(1, 10):
        count = ibi.multiply(ibi == i).getnnz()
        p = count / ibi.getnnz() * 100
        print("==", i, ":", p, "%")

    count = ibi.multiply(ibi >= 10).getnnz()
    p = count / ibi.getnnz() * 100
    print(">=", 10, ":", p, "%")


    # ---------------------- user ----------------------------
    print("ii user")
    for i in range(1, 10):
        count = iui.multiply(iui == i).getnnz()
        p = count / iui.getnnz() * 100
        print("==", i, ":", p, "%")

    count = iui.multiply(ibi >= 10).getnnz()
    p = count / ibi.getnnz() * 100
    print(">=", 10, ":", p, "%")

    # --------------------- filter out -----------------------
    n_items = ibi.shape[0]
    ibi_filter = ibi >= b_fil
    iui_filter = iui >= u_fil

    # mask all diag weight
    diag_filter_i = sp.coo_matrix(
        (np.ones(n_items), ([i for i in range(0, n_items)], [i for i in range(0, n_items)])), 
        shape=ibi.shape).tocsr()

    fil_iui = iui.multiply(iui_filter)
    fil_ibi = ibi.multiply(ibi_filter)

    # mask all diag of filtered matrix
    diag_filter_iui = fil_iui.multiply(diag_filter_i)
    diag_filter_ibi = fil_ibi.multiply(diag_filter_i)

    diag_filter_iui, diag_filter_ibi

    neighbor_ibi = fil_ibi - diag_filter_ibi.tocsc()
    neighbor_iui = fil_iui - diag_filter_iui.tocsc() 
    neighbor_ibi.getnnz(), neighbor_iui.getnnz() # -> match

    # n_ibi = normalize(neighbor_ibi, norm='l1', axis=1)
    # n_iui = normalize(neighbor_iui, norm='l1', axis=1)
    # n_ibi = neighbor_ibi
    # n_iui = neighbor_iui
    n_ibi = neighbor_ibi.tocoo()
    n_iui = neighbor_iui.tocoo()

    ibi_edge_index = torch.tensor([list(n_ibi.row), list(n_ibi.col)], dtype=torch.int64)
    iui_edge_index = torch.tensor([list(n_iui.row), list(n_iui.col)], dtype=torch.int64)

    # --------------------- saving --------------------------
    # sp.save_npz(save_path_ibi, n_ibi)
    # sp.save_npz(save_path_iui, n_iui)

    print('ibi', ibi_edge_index.shape)
    print('iui', iui_edge_index.shape)

    np.save(save_path_ibi, ibi_edge_index)
    np.save(save_path_iui, iui_edge_index)