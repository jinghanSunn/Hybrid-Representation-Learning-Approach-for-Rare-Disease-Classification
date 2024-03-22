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
from collections import Counter


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

import moco.loader
import moco.builder
from dataset import ISICDataset,ISICPsuedoDataset, ISICUnlabelDataset, ISICTestDataset, ISICPretrainDataset, ISICPretrainDatasetfoTrain
# from dataset_aptos import ISICDataset,ISICPsuedoDataset, ISICUnlabelDataset, ISICTestDataset, ISICPretrainDataset, ISICPretrainDatasetfoTrain
from dataset_papsmear import PAPSmearPsuedoDataset
from model.learner import Learner
from model.resnet import resnet12
from model.classifier import Classifier, Classifier2
from utils import save_checkpoint, plot_acc_loss, initial_classifier
from criterion import RIDELoss

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model_path', default='/dockerdata/ISIC/MOCO/moco-master/model/', type=str,
                    help='path to dataset')
parser.add_argument('--visual_dir', default='./ISIC/MOCO/moco-master/log/visual/', type=str,
                    help='path to visual')
parser.add_argument('--loss', default='origin', type=str,
                    help='origin, hard, kd')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet12',
                    # choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pe', default=10, type=int,
                    help='pretrain epoch')
parser.add_argument('--aug_num', default=1, type=int,
                    help='augment times for pretrain ')
parser.add_argument('--p_label', action='store_true', # store_true就代表着一旦有这个参数，做出动作“将其值标为True”
                    help='pseudo label train')
parser.add_argument('--n_way', default=2, type=int, help='classes for test. ')
parser.add_argument('--k_shot', default=1, type=int)
parser.add_argument('--k_query', default=15, type=int)

# distillation loss configs
parser.add_argument('-be', '--beta', type=float, default=1, help='weight balance for KD')
parser.add_argument('-al', '--alpha', type=float, default=1, help='weight for self-supervision')

# KL distillation
parser.add_argument('--kd_T', type=float, default=2, help='temperature for KD distillation')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true', # store_true就代表着一旦有这个参数，做出动作“将其值标为True”
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--pre_cla', action='store_true',
                    help='use support set feature as fc weight')

parser.add_argument('--layers', default=1, type=int, help='layers for classifier.')
parser.add_argument('--update_var', default=10, type=int, help='update var interval.')
parser.add_argument('--dataset', default='isic', type=str)



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
    if args.arch == '4conv':
        config = [
            ('conv2d', [64, 3, 3, 3, 2, 2]),
            ('relu', [True]),
            ('bn', [64]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [64, 64, 3, 3, 2, 2]),
            ('relu', [True]),
            ('bn', [64]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [64, 64, 3, 3, 2, 2]),
            ('relu', [True]),
            ('bn', [64]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [64, 64, 3, 3, 2, 2]),
            ('relu', [True]),
            ('bn', [64]),
            ('max_pool2d', [2, 2, 0]),
            ('flatten', []),
            ('fc', [64, 64])
            # ('fc', [args.moco_dim, 3136]),
        ]
        model = moco.builder.MoCo(
        # models.__dict__[args.arch],
        Learner(config),
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch)
    elif args.arch == 'resnet12':
        model = moco.builder.MoCo(
            # models.__dict__[args.arch],
            # Learner(config),
            resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.moco_dim),
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch)
    # print(model)
    extractor = model.encoder_q
    
    if args.pre_cla:
        if args.k_shot == 1:
            epoch = 70
        else:
            epoch = 130
        Classify = initial_classifier(args.k_shot, args.k_query, epoch)
    else:
        if args.layers == 1:
            Classify = Classifier(args.n_way)
        else:
            Classify = Classifier2(args.n_way)


    # for param in model.parameters():#nn.Module有成员函数parameters()
    #     print(param.requires_grad)

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



    # test(extractor_t, args)

    cudnn.benchmark = True

    # Data loading code
    traindir = args.data

    
    if args.p_label:
        if args.dataset == 'isic':
            if args.k_shot == 1:
                epoch = 70
            elif args.k_shot==3 or args.k_shot == 5:
                # epoch = 130
                epoch = 150 
            elif args.k_shot==10:
                epoch = 110
            elif args.k_shot==20:
                epoch = 150
            else:
                raise NotImplementedError
            if args.arch == '4conv':
                epoch = 150
            epoch = '{:04d}'.format(epoch)

            soft_label = np.load(f'./ISIC/MOCO/moco-master-for-public/pseudo_labels/softlabel_{args.k_shot}.npy')
            print(soft_label.shape)
            hard_label = np.argmax(soft_label, axis=-1)
            c = Counter(hard_label)
            num_img_per_cls = list(c.values())
            print("num_img_per_cls", num_img_per_cls)
            train_dataset = ISICPsuedoDataset(root=traindir, label=soft_label)
            
        else:
            seed = 222
            # seed = 444
            # seed = 555
            # seed = 666
            epoch = 1990
            #     epoch = 40
            # elif args.k_shot == 3:
            #     epoch = 120
            # elif args.k_shot == 5:
            #     epoch = 150
            epoch = '{:04d}'.format(epoch)
            soft_label = np.load(f'./data/PAPSmear/smear2005/npy/softlabel_seed{seed}_nt{epoch}_nway{args.n_way}_k{args.k_shot}.npy')
            print(soft_label.shape)
            train_dataset = PAPSmearPsuedoDataset(root=traindir, label=soft_label)
        if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    else:
        train_dataset = ISICDataset(root=traindir)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # define loss function (criterion) and optimizer
    criterion_cls = nn.CrossEntropyLoss().cuda()
    if args.p_label:
        if args.k_shot == 1:
            diveres_per_cls = [0.3320513*9000.0, 0.35115695* 9000.0, 0.31679174* 9000.0] 
        elif args.k_shot == 3:
            diveres_per_cls = [0.34690323 * 9000.0, 0.32456955 * 9000.0, 0.32852724 * 9000.0]
        elif args.k_shot == 5:
            diveres_per_cls = [0.32365558* 9000.0, 0.32892686* 9000.0, 0.3474176* 9000.0] 
        diveres_per_cls = [int(i) for i in diveres_per_cls]
        criterion_div = RIDELoss(cls_num_list=diveres_per_cls).cuda()
    else:
        criterion_div = torch.nn.MSELoss().cuda()
    print("criterion_cls", criterion_cls)
    print("criterion_div", criterion_div)

    # optimizer = torch.optim.SGD(chain(model_s.parameters(), Classify.parameters()), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay) # only update student model
    optimizer = torch.optim.SGD(chain(model.parameters(), Classify.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    alpha_start = 0.7
    all_loss_ss = []
    all_loss_div = []
    all_loss = []
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            # if args.p_label:
            #     test_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        

        alpha = alpha_start * (float(epoch)/args.epochs)
        print("alpha", alpha)

        
        # train for one epoch
        # if args.p_label:
        #     train(train_loader, model_t, model_s, extractor_t, extractor_s, criterion_cls, criterion_div, optimizer, epoch, args, test_loader=None, Classify=Classify, alpha=alpha)
        # else:
        #     train(train_loader, model_t, model_s, extractor_t, extractor_s, criterion_cls, criterion_div, optimizer, epoch, args)
        if epoch+1 % args.update_var==0:
            loss, loss_ss, loss_div, features, labels = train(train_loader, model, extractor, criterion_cls, criterion_div, optimizer, epoch, args, test_loader=None, Classify=Classify, alpha=alpha)
            vars = []
            # print(features.shape)
            # print(labels.shape)
            for i in range(3):
                tmp_feature = features[labels==i]
                var = torch.var(tmp_feature,axis=0)
                var = torch.mean(var)
                vars.extend([var.cpu().detach().item()])
            print(vars)
            diveres_per_cls = [int(i/np.sum(vars)*9000) for i in vars]
            criterion_div = RIDELoss(cls_num_list=diveres_per_cls).cuda()
        else:
            loss, loss_ss, loss_div = train(train_loader, model, extractor, criterion_cls, criterion_div, optimizer, epoch, args, test_loader=None, Classify=Classify, alpha=alpha)

        # record loss and plot fig
        all_loss.append(loss)
        all_loss_ss.append(loss_ss)
        all_loss_div.append(loss_div)
        save_name = args.model_path.split('/')[-2]
        x = np.arange(0, len(all_loss), 1)
        # print(x)
        # print(all_loss)
        plot_acc_loss(x, all_loss, path=args.visual_dir, name=f"loss = {args.alpha} * loss_ss + {args.beta} * loss_div", save_name=f'{save_name}_loss')
        plot_acc_loss(x, all_loss_ss, path=args.visual_dir, name=f"({save_name}) loss_ss", save_name=f'{save_name}_loss_ss')
        plot_acc_loss(x, all_loss_div, path=args.visual_dir, name=f"({save_name}) loss_div", save_name=f'{save_name}_loss_div')
        np.save(os.path.join(args.visual_dir, save_name+'_loss.npy'), np.array(all_loss))
        np.save(os.path.join(args.visual_dir, save_name+'_loss_ss.npy'), np.array(all_loss_ss))
        np.save(os.path.join(args.visual_dir, save_name+'_loss_div.npy'), np.array(all_loss_div))
        print("save loss npy to", os.path.join(args.visual_dir, save_name+'_loss.npy'))
        
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if epoch % 10 ==0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    # 'state_dict': model_s.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename=args.model_path + 'checkpoint_{:04d}.pth.tar'.format(epoch))
                save_checkpoint({
                    'epoch': epoch + 1,
                    # 'arch': args.arch,
                    'state_dict': Classify.state_dict(),
                    # 'state_dict': model_s.state_dict(),
                    # 'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename=args.model_path + 'checkpointClassifier_{:04d}.pth.tar'.format(epoch))


# def train(train_loader, model_t, model_s, extractor_t, extractor_s, criterion_cls, criterion_div, optimizer, epoch, args, test_loader=None, Classify=None, alpha=0, clf=None):
    
def train(train_loader, model_s, extractor_s, criterion_cls, criterion_div, optimizer, epoch, args, test_loader=None, Classify=None, alpha=0, clf=None):
    all_loss_ss = []
    all_loss_div = []
    all_loss = []

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

    # if epoch % 20 ==0:
    #     test(extractor_s, args)

    # switch to train mode
    model_s.train()
    # model_t.eval()

    end = time.time()
        
        
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # print(images[0].shape) # [8, 3, 224, 224]

        if args.gpu is not None:
            if args.p_label:
                origin = data[0].cuda(args.gpu, non_blocking=True)
                q = data[1].cuda(args.gpu, non_blocking=True)
                k = data[2].cuda(args.gpu, non_blocking=True)
                y = data[3].cuda(args.gpu, non_blocking=True)
        # print(y)
        # compute output
        output, target = model_s(im_q=q, im_k=k)
        feature_s = extractor_s(origin)
        feature_s_q = extractor_s(q)
        
        if epoch+1%args.update_var==0:
            if i==0:
                features = feature_s
                labels = y.argmax(dim=1)
            else:
                features = torch.cat((features, feature_s),dim=0)
                labels = torch.cat((labels, y.argmax(dim=1)), dim=0)
                # print(labels)

        if args.p_label:
            # with torch.no_grad():
            feature_s = Classify(feature_s) # 输出的是概率
            feature_s_q = Classify(feature_s_q)
            # feature_s_k = Classify(feature_s_k)
            feature_t = y.float()
                

        
        loss_ss = criterion_cls(output, target)
        
        if args.loss == 'origin':
            # print(feature_s, feature_t)
            loss_div = criterion_div(feature_s, feature_t) # origin
            # loss_div = criterion_div(feature_s, feature_t) + criterion_div(feature_s_q, feature_t) # + criterion_div(feature_s_k, feature_t)
        # Self-Knowledge Distillation
        elif args.loss == 'kd':
            feature_t = torch.nn.functional.one_hot(feature_t.argmax(dim=1), args.n_way)
            # loss_div = criterion_div(feature_s, (1-alpha)*feature_t+alpha*feature_s)
            # print(feature_t)
            # print(feature_s)
            loss_div = criterion_div(feature_s, (1-alpha)*feature_t+alpha*feature_s) + criterion_div(feature_s_q, (1-alpha)*feature_t+alpha*feature_s_q) # + criterion_div(feature_s, (1-alpha)*feature_t+alpha*feature_s_k)
            # print((1-alpha)*feature_t+alpha*feature_s)
        elif args.loss == 'hard':
            loss_div = criterion_cls(feature_s, feature_t.argmax(dim=1)) # hard label
            # loss_div = criterion_cls(feature_s, feature_t.argmax(dim=1)) + criterion_cls(feature_s_q, feature_t.argmax(dim=1)) # + criterion_cls(feature_s_k, feature_t.argmax(dim=1))
        else:
            raise NotImplementedError
        # print(loss_ss, loss_div)

        loss = args.alpha * loss_ss + args.beta * loss_div
        # loss = loss_div


        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        all_loss_ss.append(loss_ss.item())
        all_loss_div.append(loss_div.item())
        all_loss.append(loss.item())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_ss.update(loss_ss.item(), origin.size(0))
        losses_div.update(loss_div.item(), origin.size(0))
        losses.update(loss.item(), origin.size(0))
        top1.update(acc1[0], origin.size(0))
        top5.update(acc5[0], origin.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Classify.module.weight_norm()

        # for name, parms in Classify.named_parameters():	
        #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:', parms.grad)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    if epoch+1%args.update_var==0:
        return np.mean(all_loss), np.mean(all_loss_ss), np.mean(all_loss_div), features, labels
    return np.mean(all_loss), np.mean(all_loss_ss), np.mean(all_loss_div)

def test(model, args):
    extractor = deepcopy(model)

    # Data loading code
    testdir = args.data

    test_dataset = ISICTestDataset(root=testdir, task_num=100, n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query, selected_cls=None)

    test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(1), shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=True)

    corrects = 0
    AUC = 0
    for data in tqdm(test_loader):
        x_spt, y_spt, x_qry, y_qry = data
        x_spt = x_spt.cuda()
        y_spt = y_spt
        x_qry = x_qry.cuda()
        y_qry = y_qry

        with torch.no_grad():
            clf = LogisticRegression(random_state=0, max_iter=100).fit(extractor(x_spt.squeeze()).squeeze().detach().cpu(), y_spt.squeeze())
            q_feature = extractor(x_qry.squeeze()).squeeze().detach().cpu()
            pred_q = clf.predict(q_feature)
            prob_q = clf.predict_proba(q_feature)
            # print("pred", pred_q)
            # print(y_qry.squeeze())
            correct = (torch.eq(torch.Tensor(pred_q), y_qry.squeeze()).sum().item())/pred_q.shape[0]
            # print("acc:", correct)
            corrects += correct
        
        # print(roc_auc_score(np.eye(2)[y_qry.squeeze().cpu().numpy()], prob_q))
        AUC += roc_auc_score(np.eye(2)[y_qry.squeeze().cpu().numpy()], prob_q)
    
    # extractor = None
    
    print("accuracy:", corrects/100.0)
    print("AUC:", AUC/100.0)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
