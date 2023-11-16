# parameters need to be optimized

- `sw`: self weight in asym matrix
- `nw`: neighbor weight in asym matrix (`sw` + `nw` = 1)
- `w3`, `w4`: weight while fusing feature in item view (`w3`, `w4` in [0 ; 1])


# how to run?

sh run.sh to train on 4 datasets Youshu, clothing, food and electric
see details in run.sh file