# MSLIC

Multi-Scale-Learned-Image-Compression-V5


## Warmup

```
python -u warmup.py -exp mslicv5_mse_train --dataset ../autodl-tmp/dataset/ --epochs 200 --lambda 0.035 --metrics mse  --seed 42 --batch-size 8
```

## Train

```
python -u train.py -exp mslicv5_mse_train --dataset ../autodl-tmp/dataset/ --epochs 200 --lambda 0.035 --metrics mse  --seed 42 --batch-size 8
```

```
python -u train.py -exp mslicv5_mse_train --dataset ../autodl-tmp/dataset/ --epochs 200 --lambda 0.035 --metrics mse  --seed 42 --batch-size 8 --checkpoint ./experiments/mslicv5_mse_train/checkpoint_005.pth.tar
```

## Test

```
python -u test.py -exp mslicv5_mse_test --dataset ../autodl-tmp/dataset/ --checkpoint ./experiments/mslicv5_mse_train/checkpoint_best.pth.tar
```