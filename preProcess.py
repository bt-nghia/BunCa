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
    parser.add_argument("-iui", "--iuifil", default=4, type=int, help="iui filter")
    parser.add_argument("-ibi", "--ibifil", default=4, type=int, help="ibi filter")
    parser.add_argument("-bub", "--bubfil", default=4, type=int, help="bub filter")
    parser.add_argument("-bib", "--bibfil", default=4, type=int, help="bib filter")

    args = parser.parse_args()
    return args

if __name__=='__main__':
    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]
    ibi_fil = paras["ibifil"]
    iui_fil = paras["iuifil"]
    bub_fil = paras["bubfil"]
    bib_fil = paras["bibfil"]

    path_iui = f"datasets/{dataset_name}/iui_cooc.npz"
    path_ibi = f"datasets/{dataset_name}/ibi_cooc.npz"
    path_bub = f"datasets/{dataset_name}/bub_cooc.npz"
    path_bib = f"datasets/{dataset_name}/bib_cooc.npz"
    save_path_iui = f"datasets/{dataset_name}/n_neigh_iui"
    save_path_ibi = f"datasets/{dataset_name}/n_neigh_ibi"
    save_path_bub = f"datasets/{dataset_name}/n_neigh_bub"
    save_path_bib = f"datasets/{dataset_name}/n_neigh_bib"


    iui = load_sp_mat(path_iui)
    print("iui edge:", iui.getnnz())
    ibi = load_sp_mat(path_ibi)
    print("ibi edge:", ibi.getnnz())
    bub = load_sp_mat(path_bub)
    print("bub edge:", bub.getnnz())
    bib = load_sp_mat(path_bib)
    print("bib edge:", bib.getnnz())


    print("statistic")
    ii_b_max = int(ibi.max())
    print(f"max i-i interactions through b: {ii_b_max}")
    ii_u_max = int(iui.max())
    print(f"max i-i interactions through u: {ii_u_max}")
    bb_i_max = int(bib.max())
    print(f"max b-b interactions through i: {bb_i_max}")
    bb_u_max = int(bub.max())
    print(f"max b-b interactions through u: {bb_u_max}")

    # ---------------------- ibi ----------------------------
    print("ii bundle")
    for i in range(1, 10):
        count = ibi.multiply(ibi == i).getnnz()
        p = count / ibi.getnnz() * 100
        print("==", i, ":", p, "%")

    count = ibi.multiply(ibi >= 10).getnnz()
    p = count / ibi.getnnz() * 100
    print(">=", 10, ":", p, "%")

    # ---------------------- iui ----------------------------
    print("ii user")
    for i in range(1, 10):
        count = iui.multiply(iui == i).getnnz()
        p = count / iui.getnnz() * 100
        print("==", i, ":", p, "%")

    count = iui.multiply(iui >= 10).getnnz()
    p = count / iui.getnnz() * 100
    print(">=", 10, ":", p, "%")

    # ---------------------- bib ----------------------------
    print("bb item")
    for i in range(1, 10):
        count = bib.multiply(bib == i).getnnz()
        p = count / bib.getnnz() * 100
        print("==", i, ":", p, "%")

    count = bib.multiply(bib >= 10).getnnz()
    p = count / bib.getnnz() * 100
    print(">=", 10, ":", p, "%")

    # ---------------------- bub ----------------------------
    print("bb user")
    for i in range(1, 10):
        count = bub.multiply(bub == i).getnnz()
        p = count / bub.getnnz() * 100
        print("==", i, ":", p, "%")

    count = bub.multiply(bub >= 10).getnnz()
    p = count / bub.getnnz() * 100
    print(">=", 10, ":", p, "%")

    # --------------------- filter out -----------------------
    n_items = ibi.shape[0]
    n_bundles = bub.shape[0]
    ibi_filter = ibi >= ibi_fil
    iui_filter = iui >= iui_fil
    bub_filter = bub >= bub_fil
    bib_filter = bib >= bib_fil

    # mask all diag weight
    diag_filter_i = sp.coo_matrix(
        (np.ones(n_items), ([i for i in range(0, n_items)], [i for i in range(0, n_items)])), 
        shape=ibi.shape).tocsr()

    diag_filter_b = sp.coo_matrix(
        (np.ones(n_bundles), ([i for i in range(0, n_bundles)], [i for i in range(0, n_bundles)])),
        shape=bub.shape).tocsr()

    fil_iui = iui.multiply(iui_filter)
    fil_ibi = ibi.multiply(ibi_filter)
    fil_bub = bub.multiply(bub_filter)
    fil_bib = bib.multiply(bib_filter)

    # mask all diag of filtered matrix
    diag_filter_iui = fil_iui.multiply(diag_filter_i)
    diag_filter_ibi = fil_ibi.multiply(diag_filter_i)
    diag_filter_bub = fil_bub.multiply(diag_filter_b)
    diag_filter_bib = fil_bib.multiply(diag_filter_b)

    diag_filter_iui, diag_filter_ibi

    neighbor_ibi = fil_ibi - diag_filter_ibi.tocsc()
    neighbor_iui = fil_iui - diag_filter_iui.tocsc() 
    neighbor_bib = fil_bib - diag_filter_bib.tocsc()
    neighbor_bub = fil_bub - diag_filter_bub.tocsc()

    n_ibi = neighbor_ibi.tocoo()
    n_iui = neighbor_iui.tocoo()
    n_bub = neighbor_bub.tocoo()
    n_bib = neighbor_bib.tocoo()

    ibi_edge_index = torch.tensor([list(n_ibi.row), list(n_ibi.col)], dtype=torch.int64)
    iui_edge_index = torch.tensor([list(n_iui.row), list(n_iui.col)], dtype=torch.int64)
    bib_edge_index = torch.tensor([list(n_bib.row), list(n_bib.col)], dtype=torch.int64)
    bub_edge_index = torch.tensor([list(n_bub.row), list(n_bub.col)], dtype=torch.int64)

    # --------------------- saving --------------------------

    np.save(save_path_ibi, ibi_edge_index)
    np.save(save_path_iui, iui_edge_index)
    np.save(save_path_bub, bub_edge_index)
    np.save(save_path_bib, bib_edge_index)