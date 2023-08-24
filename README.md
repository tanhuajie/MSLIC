# MSLIC

Multi-Scale-Learned-Image-Compression-V5

## Train

```
python -u train.py -exp mslicv5_mse_train --dataset ../dataset/ --epochs 2000 --lambda 0.035 --metrics mse  --seed 42 --batch-size 8
```

```
python -u train.py -exp mslicv5_mse_train --dataset ../autodl-tmp/dataset/ --epochs 2000 --lambda 0.035 --metrics mse  --seed 42 --batch-size 8
```

## Test

```
python -u test.py -exp mslicv5_mse_test --dataset ../dataset/ --checkpoint ./experiments/mslicv5_mse_train/checkpoint_best.pth.tar
```

```
python -u test.py -exp mslicv5_mse_test --dataset ../autodl-tmp/dataset/ --checkpoint ./experiments/mslicv5_mse_train/checkpoint_best.pth.tar
```

