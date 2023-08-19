# MSLIC

Multi-Scale-Learned-Image-Compression

## Train

```
python -u train.py -exp mslic_mse_train --dataset ../dataset/ --epochs 500 --lambda 0.035 --metrics mse --batch-size 4 --seed 100 
```

## Test

```
python -u test.py -exp mslic_mse_test --dataset ../dataset/ --checkpoint ./experiments/mslic_mse_train/checkpoint_best.pth.tar
```

