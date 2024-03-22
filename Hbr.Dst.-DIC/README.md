## MoCo: Momentum Contrast for Unsupervised Visual Representation Learning


### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
nohup /usr/bin/python3 -u main_distill.py
--epochs 300 
--dist-url 'tcp://localhost:10003' 
--multiprocessing-distributed 
--world-size 1 
--rank 0 
-b 32 
--moco-k 1280 
-j 4 
./ISIC/data/ISIC/ 
--model_path ./ISIC/MOCO/model/0130distillResnet12t2NtPlKDS1/ 
--resume ./ISIC/MOCO/model/0125resnet12/checkpoint_0130.pth.tar 
--loss kd 
--p_label --n_way 3 --k_shot 1 
--kd_T 2 
> ./log/0131_0130distillResnet12t2NtPlKDS1_shot1.log &
```


### test using Logistic Regression:
```
nohup /usr/bin/python3 -u test_logis.py 
--resume ./ISIC/MO│CO/model/0130distillResnet12t2NtPlKDS1/ 
./ISIC/data/ISIC/ 
--gpu 1 
--arch resnet12 
--n_way 3 
--k_shot 1 
--select_cls 0 1 2 
--epoch 10 
--interval 10
--test_time 30
--pretrain normal
> ./log/test/0131_0130distillResnet12t2NtPlKDS5_test_logits_shot1.log &
```

### test using linear classification
```
nohup /usr/bin/python3 -u test.py 
--resume ./ISIC/MOCO/model/0125resnet12/ 
./ISIC/data/ISIC/ 
--gpu 0 
--arch resnet12 
--n_way 3 
--k_shot 1 
--lr 0.5 
--pe 20 
--epoch 130 
--pretrain normal
> ./log/test/0130_moco0125resnet12_test_randomclassifierPretrain_lr0d5_pe20_shot1.log &

nohup /usr/bin/python3 -u test.py --resume ./ISIC/MOCO/model/0125resnet12/ ./ISIC/data/ISIC/ --gpu 0 --arch resnet12 --n_way 3 --k_shot 1 --lr 0.5 --pe 20 --epoch 130 --pretrain normal > ./log/test/0130_moco0125resnet12_test_randomclassifierPretrain_lr0d5_pe20_shot1.log &
```
if only use the support set used in generating psuedo labels to pretrain LG once, set `--pretrain once`

This script uses all the default hyper-parameters as described in the MoCo v1 paper. To run MoCo v2, set `--mlp --moco-t 0.2 --aug-plus --cos`.

***Note***: for 4-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.015 --batch-size 128` with 4 gpus. We got similar results using this setting.
