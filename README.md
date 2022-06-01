# CrossCBR: Cross-view Contrastive Learning for Bundle Recommendation
This is our Pytorch implementation for the paper:

>Yunshan Ma, Yingzhi He, An Zhang, Xiang Wang, and Tat-Seng Chua (2022). CrossCBR: Cross-view Contrastive Learning for Bundle Recommendation.

Author: Dr. Yunshan Ma (yunshan.ma at u.nus.edu)

## Introduction
CrossCBR is a new recommendation model based on graph neural network and contrastive learning for bundle recommendation. By explicitly modeling the cooperative association between the item-view and bundle-view representations using an auxiliary contrastive loss, CrossCBR achieves great performance on three public bundle recommendation datasets: Youshu, NetEase, and iFashion.

## Citation 
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{CrossCBR2022,
  author    = {Yunshan Ma and
               Yingzhi He and
               An Zhang and
               Xiang Wang and
               Tat{-}Seng Chua},
  title     = {Cross-view Contrastive Learning for Bundle Recommendation},
  booktitle = {KDD},
  pages     = {},
  year      = {2022},
}
```

## Requirements
* OS: Ubuntu 18.04 or higher version
* python3.7.3
* supported(tested) CUDA versions: 10.2
* Pytorch1.8.0


## Code Structure
1. The entry script for training and evaluation is: [train.py](https://github.com/mysbupt/CrossCBR/blob/master/train.py).
2. The config file is: [config.yaml](https://github.com/mysbupt/CrossCBR/blob/master/config.yaml).
3. The script for data preprocess and dataloader: [utility.py](https://github.com/mysbupt/CrossCBR/blob/master/utility.py).
4. The model folder: [./models](https://github.com/mysbupt/CrossCBR/tree/master/models).
5. The experimental logs in tensorboard-format are saved in ./runs.
6. The experimental logs in txt-format are saved in ./log.
7. The best model and associate config file for each experimental setting is saved in ./checkpoints.

## How to run the code
1. Decompress the dataset file into the current folder: 
   '''
      tar -zxvf dataset.tgz
   '''
4. Train CrossCBR on the dataset Youshu with GPU 0: 
   '''
      python train.py -g 0 -m CrossCBR -d Youshu
   '''
   You can specify the gpu id and the used dataset by cmd line arguments, while you can tune the hyper-parameters by revising the configy file [config.yaml](https://github.com/mysbupt/CrossCBR/blob/master/config.yaml). The detailed introduction of the hyper-parameters can be seen in the config file, and you are highly encouraged to read the paper to better understand the effects of some key hyper-parameters.
