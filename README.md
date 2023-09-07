# MSLIC

Multi-Scale-Learned-Image-Compression-V5 (Base)


## Warmup

```
python -u warmup.py -exp mslicv5_mse_train --dataset ../dataset/ --epochs 160 --lambda 0.025 --metrics mse  --seed 1984 --batch-size 8
```

## Train

```
python -u train.py -exp mslicv5_mse_train --dataset ../dataset/ --epochs 160 --lambda 0.025 --metrics mse  --seed 1984 --batch-size 8
```

```
python -u train.py -exp mslicv5_mse_train --dataset ../dataset/ --epochs 160 --lambda 0.025 --metrics mse  --seed 1984 --batch-size 8 --checkpoint ./experiments/mslicv5_mse_train/checkpoint_005.pth.tar
```

## Test

```
python -u test.py -exp mslicv5_mse_test --dataset ../dataset/ --checkpoint ./experiments/mslicv5_mse_train/checkpoint_best.pth.tar
```