# MSLIC

Multi-Scale-Learned-Image-Compression-V5 (Alpha)


## Warmup

```
python -u warmup.py -exp mslic_alpha_mse_train --dataset ../dataset/ --epochs 2000 --lambda 0.035 --metrics mse  --seed 42 --batch-size 8
```

## Train

```
python -u train.py -exp mslic_alpha_mse_train --dataset ../dataset/ --epochs 2000 --lambda 0.035 --metrics mse  --seed 42 --batch-size 8
```

```
python -u train.py -exp mslic_alpha_mse_train --dataset ../dataset/ --epochs 2000 --lambda 0.035 --metrics mse  --seed 42 --batch-size 8 --checkpoint ./experiments/mslic_alpha_mse_train/checkpoint_005.pth.tar
```

## Test

```
python -u test.py -exp mslic_alpha_mse_test --dataset ../dataset/ --checkpoint ./experiments/mslic_alpha_mse_train/checkpoint_best.pth.tar
```