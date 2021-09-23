import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from copy import deepcopy
from itertools import chain
from tqdm import tqdm
# import krippendorff

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support

import moco.loader
import moco.builder
from dataset import ISICTestODataset
from model.classifier import Classifier
from model.resnet import resnet12
from utils import confidence_interval

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model_path', default='./model/', type=str,
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    # choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=222, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=1280, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# test sest configs:
parser.add_argument('--n_way', default=3, type=int,
                    help='classes for test. ')
parser.add_argument('--k_shot', default=5, type=int,
                    help='classes for test. ')

# options for moco v2
parser.add_argument('--mlp', action='store_true', 
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--load_cla', action='store_true')


def main():
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    if args.seed is not None:
        print("seed", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    main_worker(args)


def main_worker(args):
        model = moco.builder.MoCo(
            resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.moco_dim),
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch).cuda()

        Classify = Classifier(args.n_way).cuda()

        model_path = args.resume + 'checkpoint.pth.tar'
        model_path_cla = args.resume + 'checkpointClassifier.pth.tar'

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
                if args.gpu is None:
                    checkpoint = torch.load(model_path)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(model_path)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(model_path, checkpoint['epoch']))
                if args.load_cla:
                    print("=> loading checkpoint '{}'".format(model_path_cla))
                    checkpoint_cla = torch.load(model_path_cla)
                    Classify.load_state_dict({k.replace('module.',''):v for k,v in checkpoint_cla['state_dict'].items()})
                    print(f"=> loaded '{model_path_cla}'")
                success = True
            else:
                print("=> no checkpoint found at '{}'".format(model_path))
                raise NotImplementedError


        cudnn.benchmark = True

        # Data loading code
        testdir = args.data
        test_dataset = ISICTestODataset(root=testdir, k_shot=args.k_shot)

        test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=int(1), shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=True)

        model.eval()

        extractor = model.encoder_q

        corrects = []
        F1 = []
        pred_all = []
        prob_all = []
        true = []
        for data in tqdm(test_loader):
            x_qry, y_qry = data
            x_qry = x_qry.cuda()
            y_qry = y_qry.cuda()    
            with torch.no_grad():
                y_qry_pred = Classify(extractor(x_qry))
                logits_q_sm = F.softmax(y_qry_pred, dim=1)
                pred = logits_q_sm.argmax(dim=1)
                pred_all.extend(pred.cpu().detach().numpy())
                prob_all.extend(logits_q_sm.cpu().detach().numpy())
                true.extend(y_qry.cpu().detach().numpy())

        corrects = (torch.eq(torch.Tensor(pred_all), torch.Tensor(true)).sum().item())/len(test_loader)
        _, _, F1, _ = precision_recall_fscore_support(np.eye(args.n_way)[true], np.eye(args.n_way)[pred_all])

        print("accuracy:", corrects)
        print("fscore:", np.mean(F1))

    


if __name__ == '__main__':
    main()
