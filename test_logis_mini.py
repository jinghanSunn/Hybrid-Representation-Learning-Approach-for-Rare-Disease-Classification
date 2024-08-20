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
from sklearn.linear_model import LogisticRegression


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
# from torch.utils.tensorboard import SummaryWriter 
# from efficientnet_pytorch import EfficientNet

from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import cohen_kappa_score

import moco.loader
import moco.builder
# from model.resnet_mini import resnet10
from dataset_mini import MiniTestODataset, MiniPretrainDatasetfoTrain, MiniTestDataset
# from dataset_aptos import ISICDataset, ISICTestDataset, ISICTestODataset, ISICPretrainDatasetfoTrain, ISICTestODataset
from model.learner import Learner
from model.classifier import Classifier
from model.meta_origin import Meta_origin
from model.models import chainModels
from model.convNet import Convnet
from model.resnet import resnet12, resnet18
from model.resnet_cmc import resnet50
from utils import confidence_interval, plot_acc_loss

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
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    # choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
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
parser.add_argument('--pe', '--pretrain_epoch', default=20, type=int,
                    help='pretrain on support set epoch')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=111, type=int,
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
parser.add_argument('--task_num', default=1000, type=int,
                    help='number for testing tasks. ')
parser.add_argument('--n_way', default=2, type=int,
                    help='classes for test. ')
parser.add_argument('--k_shot', default=1, type=int)
parser.add_argument('--k_query', default=15, type=int)
parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
parser.add_argument('--select_cls', default=None, nargs='+', type=int)

# options for moco v2
parser.add_argument('--mlp', action='store_true', # store_true就代表着一旦有这个参数，做出动作“将其值标为True”
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


# test setting
parser.add_argument('--plot', action='store_true',
                    help='plot acc?')
parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--interval', default=10, type=int)
parser.add_argument('--test_time', default=30, type=int)
parser.add_argument('--pretrain', default='normal', type=str, help='noraml, once')
parser.add_argument('--test', default='normal', type=str, help='hard / normal / all')

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

    # log_dir = args.log_dir
    # writer = SummaryWriter(log_dir=log_dir)
    if not os.path.exists(args.visual_dir):
        os.mkdir(args.visual_dir)
        print("mkdir", args.visual_dir)

    # config = [
    #     ('conv2d', [64, 3, 3, 3, 1, 1]),
    #     ('relu', [True]),
    #     ('bn', [64]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [64, 64, 3, 3, 1, 1]),
    #     ('relu', [True]),
    #     ('bn', [64]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [64, 64, 3, 3, 2, 2]),
    #     ('relu', [True]),
    #     ('bn', [64]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [64, 64, 3, 3, 1, 1]),
    #     ('relu', [True]),
    #     ('bn', [64]),
    #     ('max_pool2d', [1, 2, 0]),
    #     ('flatten', []),
    #     # ('linear', [args.n_way, 64])
    #     ('fc', [args.moco_dim, 3136]),
    # ]
    # config = [
    #     ('conv2d', [64, 3, 3, 3, 2, 2]),
    #     ('relu', [True]),
    #     ('bn', [64]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [64, 64, 3, 3, 2, 2]),
    #     ('relu', [True]),
    #     ('bn', [64]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [64, 64, 3, 3, 2, 2]),
    #     ('relu', [True]),
    #     ('bn', [64]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [64, 64, 3, 3, 2, 2]),
    #     ('relu', [True]),
    #     ('bn', [64]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('flatten', []),
    #     # ('linear', [args.n_way, 64])
    #     ('fc', [args.moco_dim, 3136]),
    # ]
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
        ('fc', [args.moco_dim, 64])
        # ('fc', [args.moco_dim, 3136]),
    ]
    
    # if args.arch == 'resnet50' or args.arch == 'resnet18':
    #     model = moco.builder.MoCo(
    #         models.__dict__[args.arch],
    #         args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.arch)
    # elif args.arch == '4conv':
    #     model = moco.builder.MoCo(
    #         Learner(config),
    #         args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.arch)
    # elif args.arch == 'resnet12':
    #     model = moco.builder.MoCo(
    #         resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.moco_dim) ,
    #         args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.arch)
    # else:
    #     raise NotImplementedError
    # print(model)
    if args.arch == 'resnet10':
        model = moco.builder.MoCo(
            resnet10(num_classes=args.moco_dim),
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch)
    elif args.arch == 'resnet12':
        model = moco.builder.MoCo(
            resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.moco_dim),
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch)
    # elif args.arch == 'resnet18':
    #     model = moco.builder.MoCo(
    #         resnet18(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.moco_dim),
    #         args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch)
    elif args.arch == 'resnet50':
        model = moco.builder.MoCo(
            resnet50(),
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch)
    elif args.arch == '4conv':
        model = moco.builder.MoCo(
            Learner(config),
            # Convnet(),
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch)
    # elif args.arch == 'efficient':
    #     encoder = EfficientNet.from_name('efficientnet-b4')
    #     fea_in = encoder._fc.in_features
    #     encoder._fc = torch.nn.Linear(fea_in, args.moco_dim)
    #     print(encoder)
    #     model = moco.builder.MoCo(
    #         encoder,
    #         args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch)
    else:
        model = moco.builder.MoCo(
            models.__dict__[args.arch],
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch)

    model.cuda()
    
    cudnn.benchmark = True

    # Data loading code
    testdir = args.data
    pretrain_dataset = MiniPretrainDatasetfoTrain(root=testdir, task_num=args.task_num, n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query)
    if args.test == 'normal' or args.test == 'hard':
        test_dataset = MiniTestDataset(task_num=args.task_num, n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query)
    elif args.test == 'all':
        test_dataset = MiniTestODataset(n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query)
    else:
        raise NotImplementedError

    test_sampler = None

    pretrain_loader = torch.utils.data.DataLoader(
        pretrain_dataset, batch_size=int(1), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(1), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=True)

    ALL_corrects = []
    ALL_std = []
    ALL_conf = []
    ALL_AUC_macro = []
    ALL_AUC_micro = []
    ALL_AUC_sample = []
    ALL_PREICISION = []
    ALL_RECALL = []
    ALL_F1 = []

    start_ep = args.epoch # if start_qp changed, remember change the x before plot 
    for i in range(args.test_time):
        attempts = 0
        success = False
        # define loss function (criterion) and optimizer
        # criterion = nn.CrossEntropyLoss().cuda()
        model_path = args.resume + 'checkpoint_{:04d}.pth.tar'.format(start_ep)
        start_ep += args.interval
        
        # optionally resume from a checkpoint
        if model_path:
            while attempts < 20 and not success:
                try:
                    if os.path.isfile(model_path):
                        print("=> loading checkpoint '{}'".format(model_path))
                        if args.gpu is None:
                            checkpoint = torch.load(model_path)
                        else:
                            # Map model to be loaded to specified single gpu.
                            loc = 'cuda:{}'.format(args.gpu)
                            checkpoint = torch.load(model_path) # , map_location={'cuda:0':loc}
                        args.start_epoch = checkpoint['epoch']
                        # model.load_state_dict(checkpoint['state_dict'])
                        model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
                        # optimizer.load_state_dict(checkpoint['optimizer'])
                        print("=> loaded checkpoint '{}' (epoch {})"
                            .format(model_path, checkpoint['epoch']))
                        success = True
                    else:
                        print("=> no checkpoint found at '{}'".format(model_path))
                        raise NotImplementedError
                except:
                    attempts += 1
                    print("attempts:", attempts)
                    if attempts == 20:
                        break
                    time.sleep(600)
        # drop the last layer
        # if args.arch[:6] == 'resnet':
        #     extractor = torch.nn.Sequential(*(list(extractor.children())[:-1]))
        extractor = model.encoder_q
        model.eval()
        # extractor.eval()
        # if args.arch[:6] == 'resnet':
        #     extractor = torch.nn.Sequential(*(list(extractor.children())[:-1]))


        # only use support set used in generate psuedo label to pretrain
        if args.pretrain == 'once':
            for x, y in pretrain_loader:
                # print(x.shape)
                x = x.cuda()
                y = y
                clf = LogisticRegression(random_state=0, max_iter=200).fit(extractor(x.squeeze()).squeeze().detach().cpu(), y.squeeze())
            print("fit once")
        elif args.pretrain == 'none':
            pass


        corrects = []
        AUC_macro = []
        AUC_micro = []
        AUC_sample = []
        PREICISION = []
        JACCARD_sample = []
        JACCARD_micro = []
        JACCARD_macro = []
        RECALL = []
        F1 = []
        SUPPORT = []
        KAPPA = []
        pred = []
        prob = []
        true = []
        if args.test == 'normal':
            for data in tqdm(test_loader):
                x_spt, y_spt, x_qry, y_qry = data
                x_spt = x_spt.cuda()
                y_spt = y_spt
                x_qry = x_qry.cuda()
                y_qry = y_qry
                # print(x_qry.shape)
                # print(y_qry.shape)

                with torch.no_grad():
                    # normal pretrain
                    if args.pretrain == 'normal':
                        clf = LogisticRegression(random_state=0, max_iter=500).fit(extractor(x_spt.squeeze()).squeeze().detach().cpu(), y_spt.squeeze())

                    q_feature = extractor(x_qry.squeeze()).squeeze().detach().cpu()
                    pred_q = clf.predict(q_feature)
                    
                    prob_q = clf.predict_proba(q_feature)
                    # print("pred", pred_q)
                    # print(y_qry.squeeze())
                    correct = (torch.eq(torch.Tensor(pred_q), y_qry.squeeze()).sum().item())/pred_q.shape[0]
                    # print("acc:", correct)

            
                corrects.append(correct) 
                # print(roc_auc_score(np.eye(2)[y_qry.squeeze().cpu().numpy()], prob_q))
                AUC_sample.append(roc_auc_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], prob_q, average='samples'))
                AUC_micro.append(roc_auc_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], prob_q, average='micro'))
                AUC_macro.append(roc_auc_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], prob_q, average='macro'))  # samples<micro<macro=weight
                JACCARD_sample.append(jaccard_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], np.eye(args.n_way)[pred_q], average='samples'))
                JACCARD_micro.append(jaccard_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], np.eye(args.n_way)[pred_q], average='micro'))
                JACCARD_macro.append(jaccard_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], np.eye(args.n_way)[pred_q], average='macro'))
                KAPPA.append(cohen_kappa_score(y_qry.squeeze().cpu().numpy(), pred_q))
                # print(roc_auc_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], prob_q, average='samples'))
                precision, recall, fscore, _ = precision_recall_fscore_support(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], np.eye(args.n_way)[pred_q], average='weighted')
                PREICISION.append(precision) 
                RECALL.append(recall) 
                F1.append(fscore) 

            corrects = np.array(corrects)
            AUC_sample = np.array(AUC_sample)
            AUC_micro = np.array(AUC_micro)
            AUC_macro = np.array(AUC_macro)
            PREICISION = np.array(PREICISION)
            RECALL = np.array(RECALL)
            fscore = np.array(fscore)
            std = np.std(corrects, ddof = 1)
            conf_intveral = confidence_interval(std, args.task_num)
            
            print("accuracy:", np.mean(corrects))
            print("std:", std)
            print("confidence interval:",conf_intveral)
            print("AUC_sample:", np.mean(AUC_sample))
            print("AUC_micro:", np.mean(AUC_micro))
            print("AUC_macro:", np.mean(AUC_macro))
            print("precision:", np.mean(PREICISION))
            print("recall:", np.mean(RECALL))
            print("fscore:", np.mean(F1))
            print("JACCARD_sample:", np.mean(JACCARD_sample))
            print("JACCARD_micro:", np.mean(JACCARD_micro))
            print("JACCARD_macro:", np.mean(JACCARD_macro))
            print("KAPPA:", np.mean(KAPPA))
            

            ALL_corrects.append(np.mean(corrects))
            ALL_std.append(std)
            ALL_conf.append(conf_intveral)
            ALL_AUC_macro.append(np.mean(AUC_macro))
            ALL_AUC_micro.append(np.mean(AUC_micro))
            ALL_AUC_sample.append(np.mean(AUC_sample))
            ALL_PREICISION.append(np.mean(PREICISION))
            ALL_RECALL.append(np.mean(RECALL))
            ALL_F1.append(np.mean(F1))

        # # plot the results
        # if args.plot:
        #     x = np.arange(args.epoch, start_ep, args.interval)
        #     plot_acc_loss(x, ALL_corrects, path=args.visual_dir, name='accuracy', save_name = args.resume.split('/')[-2]+ f'shot{args.k_shot}' +'_accuracy')
        elif args.test == 'all':
            for data in tqdm(test_loader):
                x_qry, y_qry = data
                # x_spt = x_spt.cuda()
                # y_spt = y_spt
                x_qry = x_qry.cuda()
                y_qry = y_qry
                with torch.no_grad():
                    # if args.pretrain == 'normal':
                    #     clf = LogisticRegression(random_state=0, max_iter=200).fit(extractor(x_spt.squeeze()).squeeze().detach().cpu(), y_spt.squeeze())
                    q_feature = extractor(x_qry.float()).detach().cpu()
                    pred_q = clf.predict(q_feature)
                    prob_q = clf.predict_proba(q_feature)
                    # print("pred:", pred_q)
                    # print("y_qry:", y_qry)
                    pred.extend(pred_q)
                    prob.extend(prob_q)
                    true.extend(y_qry)
                    # print(pred_q)
                    # print(y_qry)
            print((torch.eq(torch.Tensor(pred), torch.Tensor(true)).sum().item()))
            print(pred[:200], true[:200])
            corrects = (torch.eq(torch.Tensor(pred), torch.Tensor(true)).sum().item())/len(test_loader)
            AUC_sample = roc_auc_score(np.eye(args.n_way)[true], prob, average='samples')
            AUC_micro = roc_auc_score(np.eye(args.n_way)[true], prob, average='micro')
            AUC_macro.append(roc_auc_score(np.eye(args.n_way)[true], prob, average='macro'))
            PREICISION, RECALL, F1, _ = precision_recall_fscore_support(np.eye(args.n_way)[true], np.eye(args.n_way)[pred])
            JACCARD_sample = jaccard_score(np.eye(args.n_way)[true], np.eye(args.n_way)[pred], average='samples')
            JACCARD_micro = jaccard_score(np.eye(args.n_way)[true], np.eye(args.n_way)[pred], average='micro')
            JACCARD_macro = jaccard_score(np.eye(args.n_way)[true], np.eye(args.n_way)[pred], average='macro')
            KAPPA = cohen_kappa_score(true, pred)
            # std = np.std((corrects), ddof = 1)
            # conf_intveral = confidence_interval(std, args.task_num)

            print("accuracy:", np.mean(corrects))
            # print("std:", std)
            # print("confidence interval:",conf_intveral)
            print("AUC_sample:", np.mean(AUC_sample))
            print("AUC_micro:", np.mean(AUC_micro))
            print("AUC_macro:", np.mean(AUC_macro))
            print("precision:", np.mean(PREICISION))
            print("recall:", np.mean(RECALL))
            print("fscore:", np.mean(F1))
            print("JACCARD_sample:", np.mean(JACCARD_sample))
            print("JACCARD_micro:", np.mean(JACCARD_micro))
            print("JACCARD_macro:", np.mean(JACCARD_macro))
            print("KAPPA:", np.mean(KAPPA))
        




if __name__ == '__main__':
    main()
