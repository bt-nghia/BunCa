# python train.py -w1 0.1 -w2 0.9
# python train.py -w1 0.2 -w2 0.8
# python train.py -w1 0.3 -w2 0.7
# python train.py -w1 0.4 -w2 0.6
# python train.py -w1 0.5 -w2 0.5
# python train.py -w1 0.6 -w2 0.4
# python train.py -w1 0.7 -w2 0.3
# python train.py -w1 0.8 -w2 0.2
# python train.py -w1 0.9 -w2 0.1

# python train.py -sw 0.9 -nw 0.1
# python train.py -sw 0.8 -nw 0.2
# python train.py -sw 0.7 -nw 0.3
# python train.py -sw 0.6 -nw 0.4
# python train.py -sw 0.5 -nw 0.5
# python train.py -sw 0.4 -nw 0.6
# python train.py -sw 0.3 -nw 0.7
# python train.py -sw 0.2 -nw 0.8
# python train.py -sw 0.1 -nw 0.9

# # try this instead
# python train.py -sw 1 -nw 0.9
# python train.py -sw 1 -nw 0.8
# python train.py -sw 1 -nw 0.7
# python train.py -sw 1 -nw 0.6
# python train.py -sw 1 -nw 0.5
# python train.py -sw 1 -nw 0.4
# python train.py -sw 1 -nw 0.3
# python train.py -sw 1 -nw 0.2
# python train.py -sw 1 -nw 0.1

python gene_ii_co_oc.py -d clothing
python preProcess.py -d clothing
python train.py -d clothing

python gene_ii_co_oc.py -d food
python preProcess.py -d food
python train.py -d food -w3 0.8 -w4 0.2