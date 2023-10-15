from gene_ii_asym import load_sp_mat

iui = load_sp_mat('datasets/Youshu/iui_asym.npz')
print(iui.getnnz())

ibi = load_sp_mat('datasets/Youshu/ibi_asym.npz')
print(ibi.getnnz())


ii_b_max = int(ibi.max())
print(ii_b_max)
ii_u_max = int(iui.max())
print(ii_u_max)

iiu, iib = [], []


# bundle
for i in range(1, ii_b_max):
    count = ibi.multiply(ibi == i).getnnz()
    if count != 0:
        iib.append([i, count]) 

# user
for i in range(1, ii_u_max):
    count = iui.multiply(iui == i).getnnz()
    if count != 0:
        iiu.append([i,count])

print(iiu)
print(iib)