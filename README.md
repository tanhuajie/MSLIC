# MSLIC
Multi-Scale-Learned-Image-Compression

## Train

```
python -u train.py -exp mslic_mse_train --dataset ../dataset/ --epochs 500 --lambda 0.045 --metrics mse --batch-size 8 --seed 100 
```

