# Bundle Recommendation with Item Causation-enhanced Multi-view Learning



## how to run?

- extract 3 datasets (iFashion, NetEase, Youshu) in dataset.tgz
- run `run.sh` to train on 3 datasets Youshu, iFashion, NetEase


## hyperparams

Some important hyper parameters:
- `lrs`: learning rate
- `sw`: residual connection weight
- `w1=w2`: Cohesive view weight of user/bundle
- `w3=w4`: BC sub-view user/bundle weight


## dataset

- `bundle_item.txt`: bundle-item affiliation 
- `user_item.txt`: user, item historical interaction
- `user_bundle_train/tune/test.txt`: user-bundle interaction train/valid/test set