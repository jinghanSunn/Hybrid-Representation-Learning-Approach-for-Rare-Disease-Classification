#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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
import numpy as np

import moco.loader
import moco.builder
from dataset import ISICPsuedoDataset
from model.resnet import resnet12
from model.classifier import Classifier
from criterion import DistillKL
from utils import AverageMeter, save_checkpoint, ProgressMeter, accuracy, adjust_learning_rate

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model_path', default='./model/', type=str, help='path to save model')
parser.add_argument('--visual_dir', default='./log/visual/', type=str, help='path to visual')
parser.add_argument('--savedir', default='./pseudo_labels/', type=str, help='path of pseudo labels')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet12',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training')


parser.add_argument('--p_label', action='store_true', help='pseudo-label supervised representation learning')
parser.add_argument('--n_way', default=2, type=int, help='classes for test.')
parser.add_argument('--k_shot', default=1, type=int)


# KL distillation
parser.add_argument('--kd_T', type=float, default=2, help='temperature for KD distillation')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=2048, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')






def main():
    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        print("mkdir", args.model_path)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.moco_dim),
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch)

    extractor = model.encoder_q
    
    Classify = Classifier(args.n_way)


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            Classify.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            Classify = torch.nn.parallel.DistributedDataParallel(Classify, device_ids=[args.gpu])
        else:
            model.cuda()
            Classify.cuda()
            print("distribute")
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            # extractor = torch.nn.parallel.DistributedDataParallel(extractor)
            model = torch.nn.parallel.DistributedDataParallel(model)
            Classify = torch.nn.parallel.DistributedDataParallel(Classify)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")


    # define loss function (criterion) and optimizer
    criterion_cls = nn.CrossEntropyLoss().cuda()
    criterion_div = DistillKL(args.kd_T).cuda()
    optimizer = torch.optim.SGD(chain(model.parameters(), Classify.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    traindir = args.data

    # load pseudo labels
    soft_label = np.load(args.savedir + f'/softlabel_{args.k_shot}.npy')

    train_dataset = ISICPsuedoDataset(root=traindir, label=soft_label)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    alpha_start = 0.7
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        

        alpha = alpha_start * (float(epoch)/args.epochs)
        print("alpha", alpha)

        # train for one epoch
        train(train_loader, model, extractor, criterion_cls, criterion_div, optimizer, epoch, args, Classify=Classify, alpha=alpha)

    # save models
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best=False, filename=args.model_path + 'checkpoint.pth.tar'.format(epoch))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': Classify.state_dict(),
        }, is_best=False, filename=args.model_path + 'checkpointClassifier.pth.tar'.format(epoch))


    
def train(train_loader, model_s, extractor_s, criterion_cls, criterion_div, optimizer, epoch, args, Classify=None, alpha=0):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_ss = AverageMeter('Loss_ss', ':.4e')
    losses_div = AverageMeter('Loss_div', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_ss, losses_div, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model_s.train()
    end = time.time()

    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            if args.p_label:
                q = data[1].cuda(args.gpu, non_blocking=True)
                k = data[2].cuda(args.gpu, non_blocking=True)
                y = data[3].cuda(args.gpu, non_blocking=True)
    
        # compute output
        output, target = model_s(im_q=q, im_k=k)
        feature_s_q = extractor_s(q)

        if args.p_label:
            feature_s_q = Classify(feature_s_q)
            feature_t = y.float()
        
        loss_ss = criterion_cls(output, target)
        
        # Self-Knowledge Distillation
        feature_t = torch.nn.functional.one_hot(feature_t.argmax(dim=1), args.n_way)
        loss_div = criterion_div(feature_s_q, (1-alpha)*feature_t + alpha*feature_s_q)

        loss = loss_ss + loss_div

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_ss.update(loss_ss.item(), q.size(0))
        losses_div.update(loss_div.item(), q.size(0))
        losses.update(loss.item(), q.size(0))
        top1.update(acc1[0], q.size(0))
        top5.update(acc5[0], q.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)



if __name__ == '__main__':
    main()
