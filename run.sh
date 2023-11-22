# we choose w3 w4 base on % edges in asymmatrix
# iui filtered edges >= 20% edges in iui
# ibi filtered edge ~ 1-2% edges in ibi
# we not tuning much you can tuning this coeficients to get better results


# Youshu
python gene_ii_co_oc.py -d Youshu
python preProcess.py -d Youshu -ibi 4 -iui 4
python train.py -d Youshu -w3 0.9 -w4 0.9

# NetEase
python gene_ii_co_oc.py -d NetEase
python preProcess.py -d NetEase -ibi 10 -iui 9
python train.py -d NetEase -w3 0.8 -w4 0.8

# clothing
python gene_ii_co_oc.py -d clothing
python preProcess.py -d clothing
!python train.py -d clothing -w3 0.1 -w4 0.1 -sw 0.8 -nw 0.2

# food
python gene_ii_co_oc.py -d food
python preProcess.py -d food
python train.py -d food -w3 0.9 -w4 0.1

# elctronic
python gene_ii_co_oc.py -d electronic
python preProcess.py -d electronic
python train.py -d electronic -w3 0.9 -w4 0.1
