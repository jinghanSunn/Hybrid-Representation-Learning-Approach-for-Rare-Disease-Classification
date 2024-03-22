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
import moco.builder_exampler
from dataset import ISICDataset, ISICTestDataset, ISICTestODataset, ISICPretrainDatasetfoTrain, ISICTestODatasetForCenter
# from dataset_aptos import ISICDataset, ISICTestDataset, ISICTestODataset, ISICPretrainDatasetfoTrain
from dataset_papsmear import PAPSmearPretrainDatasetfoTrain, PAPSmearTestODataset
from model.learner import Learner
from model.convNet import Convnet
from model.classifier import Classifier, Classifier2, Classifier3, Classifier4
from model.meta_origin import Meta_origin
from model.models import chainModels
from model.resnet import resnet12
from model.se_resnet import se_resnet50
from utils import confidence_interval, plot_acc_loss

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model_path', default='/dockerdata/ISIC/MOCO/moco-master/model/', type=str,
                    help='path to dataset')
parser.add_argument('--dataset', default='isic', type=str,
                    choices=['isic', 'breakhis', 'papsmear'])
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

# options for moco v2
parser.add_argument('--mlp', action='store_true', # store_true就代表着一旦有这个参数，做出动作“将其值标为True”
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--interval', default=10, type=int)
parser.add_argument('--test_time', default=30, type=int)
parser.add_argument('--load_cla', action='store_true', help='if load_cla => load classifier pretrained on distillation')
parser.add_argument('--pretrain', default='normal', type=str, help='noraml, once')
parser.add_argument('--test', default='normal', type=str, help='hard / normal')
parser.add_argument('--base', action='store_true', help="train in 4 ways")

parser.add_argument('--layers', default=1, type=int, help='layers for classifier. ')

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
    #     ('fc', [args.moco_dim, 768]),
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
    


    start_ep = args.epoch # if start_qp changed, remember change the x before plot 
    for i in range(args.test_time):
        attempts = 0
        success = False

        if args.arch == 'resnet12':
            model = moco.builder.MoCo(
                resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.moco_dim),
                args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch).cuda()
        elif args.arch == '4conv':
            model = moco.builder.MoCo(
                Learner(config),
                # Convnet(),
                args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch).cuda()
        # elif args.arch == 'efficient':
        #     encoder = EfficientNet.from_name('efficientnet-b4')
        #     fea_in = encoder._fc.in_features
        #     encoder._fc = torch.nn.Linear(fea_in, args.moco_dim)
        #     print(encoder)
        #     model = moco.builder.MoCo(
        #         encoder,
        #         args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch)
        elif args.arch == 'senet':
            backbone = se_resnet50(num_classes=args.moco_dim, pretrained=False)
            model = moco.builder.MoCo(
                backbone,
                args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch).cuda()
        else:
            model = moco.builder.MoCo(
                models.__dict__[args.arch],
                args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, arch=args.arch)

        # Classify = Classifier(args.n_way)
        if args.layers == 1:
            Classify = Classifier(args.n_way).cuda()
        else:
            Classify = Classifier2(args.n_way).cuda()

        model_path = args.resume + 'checkpoint_{:04d}.pth.tar'.format(start_ep)
        model_path_cla = args.resume + 'checkpointClassifier_{:04d}.pth.tar'.format(start_ep)
        start_ep += args.interval

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()

        # optionally resume from a checkpoint
        if args.resume:
            while attempts < 20 and not success:
                try:
                    if os.path.isfile(model_path):
                        print("=> loading checkpoint '{}'".format(model_path))
                        if args.gpu is None:
                            checkpoint = torch.load(model_path)
                        else:
                            # Map model to be loaded to specified single gpu.
                            loc = 'cuda:{}'.format(args.gpu)
                            checkpoint = torch.load(model_path) # , map_location=loc
                        args.start_epoch = checkpoint['epoch']
                        # model.load_state_dict(checkpoint['state_dict'])
                        model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
                        # optimizer.load_state_dict(checkpoint['optimizer'])
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
                except:
                    attempts += 1
                    print("attempts:", attempts)
                    if attempts == 20:
                        break
                    time.sleep(600)

        cudnn.benchmark = True

        # Data loading code
        testdir = args.data
        if args.dataset == 'isic':
            pretrain_dataset = ISICPretrainDatasetfoTrain(root=testdir, task_num=args.task_num, n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query)
            if args.test == 'normal' or args.test == 'hard':
                test_dataset = ISICTestDataset(root=testdir, task_num=args.task_num, n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query, selected_cls=[0,1,2], test=args.test)
            elif args.test == 'all':
                if args.base:
                    test_dataset = ISICTestODatasetForCenter(root=testdir, task_num=args.task_num, n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query)
                else:
                    test_dataset = ISICTestODataset(root=testdir, task_num=args.task_num, n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query, selected_cls=[0,1,2])
            else:
                raise NotImplementedError
        elif args.dataset == 'papsmear':
            pretrain_dataset = PAPSmearPretrainDatasetfoTrain(root=testdir, task_num=args.task_num, n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query)
            test_dataset = PAPSmearTestODataset(root=testdir, task_num=args.task_num, n_way=args.n_way, k_shot=args.k_shot)
        test_sampler = None
        pretrain_loader = torch.utils.data.DataLoader(
            pretrain_dataset, batch_size=int(1), shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=int(1), shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=True)

        model.eval()

        if args.pretrain == 'once':
            extractor = deepcopy(model.encoder_q)
            Classifi = deepcopy(Classify)
            optimizer = torch.optim.SGD(Classifi.parameters(), lr=args.lr,) # only finetune classifier
            # print("training from scrach")
            # optimizer = torch.optim.Adam(chain(extractor.parameters(), Classifi.parameters()), lr=args.lr,)
            for x, y in pretrain_loader:
                x = x.cuda()
                y = y.cuda()
                for  i in range(args.pe):
                    with torch.no_grad():
                        feature = extractor(x.squeeze()) # [2, 128]
                    logits = Classifi(feature) 
                    loss = criterion(logits, y.squeeze())
                    print(loss)

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # Classifi.weight_norm()
        elif args.pretrain == 'none':
            extractor = model.encoder_q
            Classifi = Classify

        # corrects = 0
        # AUC = 0
        # pred_all = []
        # true = []
        # flag = 0

        # Classifi = deepcopy(Classify)
        corrects = []
        AUC_macro = []
        AUC_micro = []
        AUC_sample = []
        PREICISION = []
        RECALL = []
        F1 = []
        pred_all = []
        prob_all = []
        JACCARD_sample = []
        JACCARD_micro = []
        JACCARD_macro = []
        KAPPA = []
        true = []
        if args.test == 'normal':
            for data in tqdm(test_loader):
                x_spt, y_spt, x_qry, y_qry = data
                x_spt = x_spt.cuda()
                y_spt = y_spt.cuda()
                x_qry = x_qry.cuda()
                y_qry = y_qry.cuda()

                if args.pretrain == 'normal':
                    # # deepcopy model for each task
                    extractor = deepcopy(model.encoder_q)
                    Classifi = deepcopy(Classify)
                    # optimizer = torch.optim.SGD(params=chain(extractor.parameters(), Classifi.parameters()), lr=args.lr,) # finetune all model
                    optimizer = torch.optim.SGD(Classifi.parameters(), lr=args.lr,) # only finetune classifier


                    # pretrain
                    for  i in range(args.pe):
                        with torch.no_grad():
                            feature = extractor(x_spt.squeeze()) # [2, 128]
                        logits = Classifi(feature) 
                        loss = criterion(logits, y_spt.squeeze())
                        # flag = 1
                        

                        # compute gradient and do SGD step
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # test
                        # if i % 1==0:
                        #     # print(i,"loss:",loss)
                        #     with torch.no_grad():
                        #         y_qry_pred = Classifi(extractor(x_qry.squeeze()))
                        #         logits_q_sm = F.softmax(y_qry_pred, dim=1)
                        #         correct = (torch.eq(logits_q_sm.argmax(dim=1), y_qry.squeeze()).sum().item())/y_qry_pred.shape[0]
                        #         print(i+1, "acc:", correct)

                # with torch.no_grad():
                #     y_qry_pred = Classify(model.encoder_q(x_qry.squeeze()))
                #     logits_q_sm = F.softmax(y_qry_pred, dim=1)
                #     correct = (torch.eq(logits_q_sm.argmax(dim=1), y_qry.squeeze()).sum().item())/y_qry_pred.shape[0]
                #     print("acc:", correct)
                #     corrects += correct
                with torch.no_grad():
                    y_qry_pred = Classifi(extractor(x_qry.squeeze()))
                    logits_q_sm = F.softmax(y_qry_pred, dim=1)
                    correct = (torch.eq(logits_q_sm.argmax(dim=1), y_qry.squeeze()).sum().item())/y_qry_pred.shape[0]
                    # print("acc:", correct)
                    # corrects += correct
                corrects.append(correct) 
                # print(roc_auc_score(np.eye(2)[y_qry.squeeze().cpu().numpy()], prob_q))
                AUC_sample.append(roc_auc_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], logits_q_sm.cpu(), average='samples'))
                AUC_micro.append(roc_auc_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], logits_q_sm.cpu(), average='micro'))
                AUC_macro.append(roc_auc_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], logits_q_sm.cpu(), average='macro'))  # samples<micro<macro=weight
                JACCARD_sample.append(jaccard_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], np.eye(args.n_way)[logits_q_sm.argmax(dim=1).cpu()], average='samples'))
                JACCARD_micro.append(jaccard_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], np.eye(args.n_way)[logits_q_sm.argmax(dim=1).cpu()], average='micro'))
                JACCARD_macro.append(jaccard_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], np.eye(args.n_way)[logits_q_sm.argmax(dim=1).cpu()], average='macro'))
                KAPPA.append(cohen_kappa_score(y_qry.squeeze().cpu().numpy(), logits_q_sm.argmax(dim=1).cpu()))
                # print(roc_auc_score(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], prob_q, average='samples'))
                precision, recall, fscore, _ = precision_recall_fscore_support(np.eye(args.n_way)[y_qry.squeeze().cpu().numpy()], np.eye(args.n_way)[logits_q_sm.argmax(dim=1).cpu()], average='weighted')
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


        elif args.test == 'all':
            for j, data in enumerate(tqdm((test_loader))):
                x_spt, y_spt, x_qry, y_qry = data
                x_spt = x_spt.cuda()
                y_spt = y_spt.cuda()
                x_qry = x_qry.cuda()
                y_qry = y_qry.cuda()    
                with torch.no_grad():
                    y_qry_pred = Classifi(extractor(x_qry))
                    logits_q_sm = F.softmax(y_qry_pred, dim=1)
                    pred = logits_q_sm.argmax(dim=1)
                    # if pred.cpu().detach().numpy()!=y_qry.cpu().detach().numpy():
                    #     print(j, "pred:", pred.cpu().detach().numpy(), 'gt:', y_qry.cpu().detach().numpy())
                    # print("acc:", correct)
                    pred_all.extend(pred.cpu().detach().numpy())
                    prob_all.extend(logits_q_sm.cpu().detach().numpy())
                    true.extend(y_qry.cpu().detach().numpy())
                        # print(pred, true)
            # print(pred_all)
            # # prob_all = np.array(prob_all)
            # # print(prob_all.shape)
            # print(true)
            # print(torch.eq(torch.Tensor(pred_all), torch.Tensor(true)))
            # print(torch.eq(torch.Tensor(pred_all), torch.Tensor(true)).sum().item())
            # print("base acc:", (torch.eq(torch.Tensor(pred_all[-200:]), torch.Tensor(true[-200:])).sum().item())/200)
            corrects = (torch.eq(torch.Tensor(pred_all), torch.Tensor(true)).sum().item())/len(test_loader)
            # print(torch.eq(torch.Tensor(pred_all), torch.Tensor(true)))
            # print((torch.eq(torch.Tensor(pred_all), torch.Tensor(true)).sum().item()))
            # print(len(test_loader))
            # AUC_sample = roc_auc_score(np.eye(args.n_way)[true], prob_all, average='samples')
            # AUC_micro = roc_auc_score(np.eye(args.n_way)[true], prob_all, average='micro')
            # AUC_macro.append(roc_auc_score(np.eye(args.n_way)[true], prob_all, average='macro'))
            PREICISION, RECALL, F1, _ = precision_recall_fscore_support(np.eye(args.n_way)[true], np.eye(args.n_way)[pred_all])
            JACCARD_sample = jaccard_score(np.eye(args.n_way)[true], np.eye(args.n_way)[pred_all], average='samples')
            JACCARD_micro = jaccard_score(np.eye(args.n_way)[true], np.eye(args.n_way)[pred_all], average='micro')
            JACCARD_macro = jaccard_score(np.eye(args.n_way)[true], np.eye(args.n_way)[pred_all], average='macro')
            KAPPA = cohen_kappa_score(true, pred_all)
            # reliability_data = [np.eye(args.n_way)[pred_all],np.eye(args.n_way)[true]]
            # kappa = krippendorff.alpha(reliability_data=reliability_data)
            # for i in np.unique(true):
                # print(i)
                # print(torch.sum((torch.Tensor(pred_all)==i)))
                # print(torch.sum(torch.Tensor(true)==i))
                # print(torch.sum((torch.Tensor(pred_all)==i))/torch.sum(torch.Tensor(true)==i))
            #[110, 322, 137]
            if args.dataset == 'isic':
                # index1 = 115 - args.k_shot
                # index2 = 327 - args.k_shot
                # index3 = 142 - args.k_shot
                index1 = 228 - args.k_shot
                index2 = 97 - args.k_shot
                index3 = 74 - args.k_shot
            elif args.dataset == 'papsmear':
                # Counter({2: 97, 0: 73, 1: 69})
                index1 = 97 - args.k_shot
                index2 = 73 - args.k_shot
                index3 = 69 - args.k_shot
            print(index1,index2,index3)
            acc_1 = (torch.eq(torch.Tensor(pred_all)[:index1], torch.Tensor(true)[:index1]).sum().item())/float(index1)
            acc_2 = (torch.eq(torch.Tensor(pred_all)[index1:index1+index2], torch.Tensor(true)[index1:index1+index2]).sum().item())/float(index2)
            acc_3 = (torch.eq(torch.Tensor(pred_all)[index1+index2:], torch.Tensor(true)[index1+index2:]).sum().item())/float(index3)
            print(acc_1)
            print(acc_2)
            print(acc_3)
            print("avg:", (acc_1+acc_2+acc_3)/3)


            print("accuracy:", corrects)
            # print("AUC_sample:", np.mean(AUC_sample))
            # print("AUC_micro:", np.mean(AUC_micro))
            # print("AUC_macro:", np.mean(AUC_macro))
            print("precision:", np.mean(PREICISION))
            print("recall:", np.mean(RECALL))
            print("fscore:", np.mean(F1))
            print("JACCARD_sample:", np.mean(JACCARD_sample))
            print("JACCARD_micro:", np.mean(JACCARD_micro))
            print("JACCARD_macro:", np.mean(JACCARD_macro))
            print("KAPPA:", np.mean(KAPPA))

        np.save(f'./ISIC/Pred/ISIC_Hybrid_shot{args.k_shot}.npy', np.array(pred_all))
        print('save to', f'./ISIC/Pred/ISIC_Hybrid_shot{args.k_shot}.npy')

        del extractor
        del Classifi
        del model
        del Classify
    


if __name__ == '__main__':
    main()
