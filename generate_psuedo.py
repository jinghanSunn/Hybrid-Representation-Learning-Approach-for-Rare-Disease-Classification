import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from copy import deepcopy
from tqdm import tqdm
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
from collections import Counter

import moco.loader
import moco.builder
from dataset import ISICUnlabelDataset, ISICPretrainDataset
from model.resnet import resnet12

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--n_way', default=3, type=int)
parser.add_argument('--k_shot', default=1, type=int)
parser.add_argument('--k_query', default=1, type=int)
parser.add_argument('--augnum', default=5, type=int)
parser.add_argument('--seed', default=111, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--arch', default='resnet12', type=str)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--resume', default='.model/checkpoint.pth.tar', type=str)
parser.add_argument('--datadir', default='./data/ISIC/', type=str)
parser.add_argument('--savedir', default='./pseudo_labels/', type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

seed = args.seed
print("seed", seed)

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

resume = args.resume
gpu = args.gpu
traindir = args.datadir
n_way = args.n_way
k_shot = args.k_shot 
k_query = args.k_query
aug_num = args.augnum

workers = 4

model = moco.builder.MoCo(
        resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=128), 
        128, 1280, 0.999, 0.07, arch=args.arch).cuda()
extractor = model.encoder_q

if resume:
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        if gpu is None:
            checkpoint = torch.load(resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

extractor = model.encoder_q

test_dataset = ISICPretrainDataset(root=traindir, n_way=n_way, k_shot=k_shot, k_query=k_query, aug_num=aug_num)

test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=True)

# save pretrain task
task = test_dataset.get_pretrain_task()
np.save(args.savedir+f"pretrain_task_shot{k_shot}", task)

# pretrain with logistic regression
features = []
labels = []
for data in test_loader:
    x_spt, y_spt, x_qry, y_qry = data
    x_spt = x_spt.cuda()
    y_spt = y_spt
    x_qry = x_qry.cuda()
    y_qry = y_qry
    with torch.no_grad():
        feature = extractor(x_spt.squeeze()).squeeze().detach().cpu()

        clf = LogisticRegression(random_state=0, max_iter=100).fit(feature, y_spt.squeeze())
        q_feature = extractor(x_qry.squeeze()).squeeze().detach().cpu()
        pred_q = clf.predict(q_feature)
        prob_q = clf.predict_proba(q_feature)
        features.extend(feature.numpy())
        labels.extend(y_spt.numpy())

        correct = (torch.eq(torch.Tensor(pred_q), y_qry.squeeze()).sum().item())/pred_q.shape[0]
        print("pretrain acc:", correct)


print("Generate psuedo label")
UnlabelledDataset = ISICUnlabelDataset(traindir)
UnlabelledDataLoader = torch.utils.data.DataLoader(
    UnlabelledDataset, batch_size=16, shuffle=False,
    num_workers=workers, pin_memory=True, drop_last=False)
soft_label = []
hard_label = []
count = 0
for img in tqdm(UnlabelledDataLoader):
    img = img.cuda()
    with torch.no_grad():
        prob_q = clf.predict_proba(extractor(img).cpu())
        pred_q = clf.predict(extractor(img).cpu())
    soft_label.extend(prob_q)
    hard_label.extend(pred_q)
soft_label = np.array(soft_label)
hard_label = np.array(hard_label)


np.save(args.savedir+f'softlabel_{k_shot}.npy', soft_label)
np.save(args.savedir+f'hardlabel_{k_shot}.npy', hard_label)

