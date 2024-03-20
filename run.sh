# iFashion
python gene_ii_co_oc.py -d iFashion
python preProcess.py -d iFashion -iui 15 -ibi 2
python train.py -d iFashion -m LightGCN


# NetEase
python gene_ii_co_oc.py -d NetEase
python preProcess.py -d NetEase -ibi 10 -iui 9
python train.py -d NetEase -m LightGCN


# Youshu
python gene_ii_co_oc.py -d Youshu
python preProcess.py -d Youshu -ibi 4 -iui 4
python train.py -d Youshu -m LightGCN
