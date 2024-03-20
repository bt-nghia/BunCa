# iFashion
python gene_ii_co_oc.py -d iFashion
python preProcess.py -d iFashion -iui 15 -ibi 2
python train.py -d iFashion -w1 0.6 -w2 0.6 -w3 0.9 -w4 0.9 -sw 0.9 -nw 0.1 -m LightGCN


# NetEase
python gene_ii_co_oc.py -d NetEase
python preProcess.py -d NetEase -ibi 10 -iui 9
python train.py -d NetEase -w1 1 -w2 1 -w3 0.8 -w4 0.8 -sw 0 -nw 1 -m LightGCN


# Youshu
python gene_ii_co_oc.py -d Youshu
python preProcess.py -d Youshu -ibi 4 -iui 4
python train.py -d Youshu -w1 1 -w2 1 -w3 0.9 -w4 0.9 -sw 0 -nw 1 -m LightGCN
