from gene_ii_asym import load_sp_mat

iui = load_sp_mat('datasets/Youshu/iui_asym.npz')
print(iui.getnnz())
ibi = load_sp_mat('datasets/Youshu/ibi_asym.npz')
print(ibi.getnnz())


ii_b_max = int(ibi.max())
print(ii_b_max)
ii_u_max = int(iui.max())
print(ii_u_max)


# bundle
print("statistic")
print("ii bundle")
for i in range(1, 10):
    count = ibi.multiply(ibi == i).getnnz()
    p = count / ibi.getnnz() * 100
    print("==", i, ":", p, "%")

count = ibi.multiply(ibi >= 10).getnnz()
p = count / ibi.getnnz() * 100
print(">=", 10, ":", p, "%")


# user
print("ii user")
for i in range(1, 10):
    count = iui.multiply(iui == i).getnnz()
    p = count / iui.getnnz() * 100
    print("==", i, ":", p, "%")

count = iui.multiply(ibi >= 10).getnnz()
p = count / ibi.getnnz() * 100
print(">=", 10, ":", p, "%")


from sklearn.preprocessing import normalize
import scipy.sparse as sp
import numpy as np

n_items = ibi.shape[0]
ibi_filter = ibi >= 2
iui_filter = iui >= 2

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

n_ibi = neighbor_ibi
n_iui = neighbor_iui

sp.save_npz("datasets/Youshu/n_neigh_ibi.npz", n_ibi)
sp.save_npz("datasets/Youshu/n_neigh_iui.npz", n_iui)