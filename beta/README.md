# MSLIC

Multi-Scale-Learned-Image-Compression-V6 (BETA)


## Warmup (NVIDIA GeForce RTX 4090)

```
python -u warmup.py -exp mslicv6_mse_train --dataset ../dataset/ --epochs 300 --lambda 0.035 --metrics mse  --seed 42 --batch-size 8
```

## Train (NVIDIA GeForce RTX 4090)

```
python -u train.py -exp mslicv6_mse_train --dataset ../dataset/ --epochs 300 --lambda 0.035 --metrics mse  --seed 42 --batch-size 8
```

```
python -u train.py -exp mslicv6_mse_train --dataset ../dataset/ --epochs 300 --lambda 0.035 --metrics mse  --seed 42 --batch-size 8 --checkpoint ./experiments/mslicv6_mse_train/checkpoint_005.pth.tar
```

## Test

```
python -u test.py -exp mslicv6_mse_test --dataset ../dataset/ --checkpoint ./experiments/mslicv6_mse_train/checkpoint_best.pth.tar
```