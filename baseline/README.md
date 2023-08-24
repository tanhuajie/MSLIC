# MSLIC

Multi-Scale-Learned-Image-Compression-V-4

## Train

```
python -u train.py -exp mslicv4_mse_train --dataset ../dataset/ --epochs 500 --lambda 0.035 --metrics mse  --seed 42 --batch-size 8
```

## Test

```
python -u test.py -exp mslicv4_mse_test --dataset ../dataset/ --checkpoint ./experiments/mslicv4_mse_train/checkpoint_best.pth.tar
```

