# Bundle Recommendation with Item Causation-enhanced Multi-view Learning
This is the Pytorch implementation for paper " Bundle Recommendation with Item Causation-enhanced Multi-view Learning "


## How to run?

- extract 3 datasets (iFashion, NetEase, Youshu) in dataset.tgz
- run `run.sh` to train on 3 datasets Youshu, iFashion, NetEase


## Hyperparams

Some important hyper parameters:
- `lrs`: learning rate
- `sw`: residual connection weight
- `w1/w2`: Cohesive view weight of user/bundle
- `w3/w4`: BC sub-view user/bundle weight


## Dataset

- `bundle_item.txt`: bundle-item affiliation 
- `user_item.txt`: user, item historical interaction
- `user_bundle_train/tune/test.txt`: user-bundle interaction train/valid/test set


## Requirements

- torch == 2.0.1
- scipy == 1.11.2
- torch-geometric == 2.3.1
- tensorboardX == 2.6.2.2