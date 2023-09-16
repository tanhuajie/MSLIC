# MSLIC

Multi-Scale-Learned-Image-Compression-V5 (Base)


## Warmup

```
python -u warmup.py -exp mslicv5_mse_train --dataset ../dataset/ --epochs 240 --lambda 0.025 --metrics mse  --seed 1984 --batch-size 16 --gpu_id 0 --patch-size 256 256 
```

```
nohup python -u warmup.py -exp mslicv5_mse_train --dataset ../dataset/ --epochs 240 --lambda 0.025 --metrics mse --seed 1984 --batch-size 16 --gpu_id 0 --patch-size 256 256 > training_s1.log 2>&1 &
```

## Train

```
python -u train.py -exp mslicv5_mse_train --dataset ../dataset/ --epochs 240 --lambda 0.025 --metrics mse  --seed 1984 --batch-size 16 --gpu_id 0 --patch-size 256 256 --checkpoint ./experiments/mslicv5_mse_train/checkpoint_002.pth.tar
```

```
python -u train.py -exp mslicv5_mse_train --dataset ../dataset/ --epochs 160 --lambda 0.025 --metrics mse  --seed 1984 --batch-size 16 --gpu_id 0,1 --patch-size 512 512 --checkpoint ./experiments/mslicv5_mse_train/checkpoint_240.pth.tar
```

```
nohup python -u train.py -exp mslicv5_mse_train --dataset ../dataset/ --epochs 240 --lambda 0.025 --metrics mse  --seed 1984 --batch-size 16 --gpu_id 0 --patch-size 256 256 --checkpoint ./experiments/mslicv5_mse_train/checkpoint_002.pth.tar > training_s2.log 2>&1 &
```

```
nohup python -u train.py -exp mslicv5_mse_train --dataset ../dataset/ --epochs 160 --lambda 0.025 --metrics mse  --seed 1984 --batch-size 16 --gpu_id 0,1 --patch-size 512 512 --checkpoint ./experiments/mslicv5_mse_train/checkpoint_240.pth.tar > training_s3.log 2>&1 &
```


## Test

```
python -u test.py -exp mslicv5_mse_test --dataset ../dataset/ --checkpoint ./experiments/mslicv5_mse_train/checkpoint_best.pth.tar
```