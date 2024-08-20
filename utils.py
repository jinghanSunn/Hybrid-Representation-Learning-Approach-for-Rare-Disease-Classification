import numpy as np
import torch
import shutil

import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import ISICDistillDataset, ISICPretrainDatasetfoTrain
from model.classifier import Classifier
from collections import Counter

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

import moco.loader
import moco.builder
from model.resnet import resnet12

def adjust_learning_rate(epoch, args, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
    if steps > 0:
        new_lr = args.lr * (args.lr_decay_rate ** steps)
        print("new_lr:", new_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print("save model to", filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



def confidence_interval(std, n):
    return 1.92 * std / np.sqrt(n)


def plot_acc_loss(x, y, path = None, name='accuracy', save_name='accuracy'):
    # host = host_subplot(111)  # row=1 col=1 first pic
    # plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    # par1 = host.twinx()   # 共享x轴
 
    # # set labels
    # host.set_xlabel("epochs")
    # host.set_ylabel("test-accuracy")
    # # par1.set_ylabel("test-accuracy")
 
    # # plot curves
    # p1, = host.plot(x, acc, label="accuracy")
    # p2, = par1.plot(range(len(acc)), acc, label="accuracy")
 
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    # host.legend(loc=5)
 
    # set label color
    # host.axis["left"].label.set_color(p1.get_color())
    # par1.axis["right"].label.set_color(p2.get_color())
 
    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])
 
    # plt.draw()
    # plt.show()

    fig, ax1 = plt.subplots()
    lns1 = ax1.plot(x, y, label=name)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel(name)

    plt.savefig(os.path.join(path, f"{save_name}.png"))
    print(f"save {save_name} fig to", os.path.join(path, f"{save_name}.png"))
    plt.close('all')



def openImage(x, path):
    return Image.open(path + x + '.jpg').convert('RGB')


def calOriginMean():
    root = './ISIC/data/ISIC/'
    path = 'ISIC2018_Task3_Training_Input/'

    x = []
    for i in range(7):
        x.append(np.load(os.path.join(root, f'class{i}.npy')))

    for j in range(1, 7):
        mean = []
        std = []
        for img_name in tqdm(x[j]):
            img = openImage(img_name, root+path)
            mean.append(np.mean(img))
            std.append(np.std(img))
        print(j)
        print("mean:", np.mean(mean))
        print("max mean", np.max(mean))
        print("min mean", np.min(mean))
        print("std:", np.mean(std))
        print("max std:", np.max(std))
        print("min std", np.min(std))

    '''
    0
    mean: 160.03025215787002
    max mean 231.770587654321
    min mean 84.62098395061729
    std: 43.488797293323216
    max std: 89.7438687136049
    min std 10.392343580805482

    1
    mean: 153.5607636728672
    max mean 224.06382592592593
    min mean 63.62614444444444
    std: 41.59414316085248
    max std: 83.83438261801908
    min std 17.22959852627346

    2
    mean: 154.95032959031218
    max mean 212.60316543209876
    min mean 99.86593827160493
    std: 33.73829367666433
    max std: 77.19571354447156
    min std 12.11372541466443

    3
    mean: 169.64586942565757
    max mean 215.64253950617285
    min mean 123.46024691358025
    std: 33.10940757428141
    max std: 57.05711735100219
    min std 15.868418148392177

    4
    mean: 167.41976825612565
    max mean 202.05185679012345
    min mean 123.94952962962962
    std: 34.25405632045517
    max std: 63.67835777239099
    min std 19.05827120001288

    5
    mean: 171.1482450064851
    max mean 211.0342925925926
    min mean 109.18143580246914
    std: 30.28646915692229
    max std: 66.6598183481614
    min std 10.61642341204884

    6
    mean: 169.8432323508955
    max mean 229.4353987654321
    min mean 122.24032469135803
    std: 36.26562653896307
    max std: 72.98452749915285
    min std 16.467255660354713
    '''


def calFeatureMean():
    root = './ISIC/data/ISIC/'
    
    # model 
    model = moco.builder.MoCo(resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=128), 128, 1280, 0.999, 0.07).cuda()
    extractor = model.encoder_q
    
    
    
    dataset = ISICDistillDataset(root, [0,1,2,3,4,5,6])
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, sampler=None, drop_last=True)

    resume = './ISIC/MOCO/model/resnet12/checkpoint.pth.tar'
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            # if gpu is None:
            checkpoint = torch.load(resume)
            # model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()}, strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    means = torch.zeros((7,1))
    means = np.array(means)
    for i in range(7):
        means[i] = list(means[i])
    means = means.tolist()
    # print(means)
    # means = np.array(means)
    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.cuda()
        with torch.no_grad():
            feature = extractor(x)
        mean = torch.mean(feature, dim=1)
        print(y.squeeze().detach().cpu())
        print(mean)
        # tmp = mean.item()
        # print(tmp)
        # print(y.squeeze().detach().cpu())
        means[y.squeeze().detach().cpu().item()].extend([mean.item()])
        # print(means)
        # if i==4:
            # break
            
    # print(np.array(means))
    for i in range(7):
        print(i, ":", np.mean(means[i]))

    '''
    0 : -0.0010224149473258143                                                                                                                                           
    1 : -0.018509603882353932                                                                                                                                            
    2 : -0.010291556300150908                                                                                                                                            
    3 : 0.0031690366911412827                                                                                                                                            
    4 : -0.004567170198353734                                                                                                                                            
    5 : -0.0017370448574351453                                                                                                                                           
    6 : -0.010834222190550991
    '''

def parse_acc():
    fp = open("tmp.log")
    result = []
    for line in fp.readlines():
        start = line.find('acc')
        if start!=-1:
            # print(line[(start+4):])
            result.append(float(line[(start+4):]))
    print("final", np.mean(result))
    fp.close()

def initial_classifier(k_shot=1, k_query=15, epoch=70):
    gpu = 1
    epoch = '{:04d}'.format(epoch)
    resume = f'./ISIC/MOCO/model/0125resnet12/checkpoint_{epoch}.pth.tar'

    # model
    model = moco.builder.MoCo(
            # models.__dict__[args.arch],
            # Learner(config),
            resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=128),
            128, 1280, 0.999, 0.07, arch='resnet12').cuda()
        # print(model)
    extractor = model.encoder_q

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            if gpu is None:
                checkpoint = torch.load(resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(resume, map_location={'cuda:0':loc})
            start_epoch = checkpoint['epoch']
            # model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    extractor = model.encoder_q
    
    Classifi = Classifier(3)
    pretrain_dataset = ISICPretrainDatasetfoTrain(root='./ISIC/data/ISIC/', task_num=1, n_way=3, k_shot=k_shot, k_query=k_query)
    with torch.no_grad():
        for x, y in pretrain_dataset:
            x.cuda()
            output = extractor(x.squeeze())
            output_stack = output
            target_stack = y.squeeze()

    new_weight = torch.zeros(3, 128)
    for i in range(3):
        tmp = output_stack[target_stack == i].mean(0)
        new_weight[i] = tmp / tmp.norm(p=2)
    weight = new_weight.cuda()
    Classifi.layer[1].weight.data= weight
    del model
    return Classifi

def initial_classifier_oct(k_shot=1, k_query=15, epoch=70, args=None):
    gpu = 1
    epoch = '{:04d}'.format(epoch)
    resume = f'./ISIC/MOCO/model/resnet12/checkpoint_{epoch}.pth.tar'

    # model
    model = moco.builder.MoCo(
            # models.__dict__[args.arch],
            # Learner(config),
            resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=128),
            128, 256, 0.999, 0.07, arch='resnet12').cuda()
        # print(model)
    extractor = model.encoder_q

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            if gpu is None:
                checkpoint = torch.load(resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(resume, map_location={'cuda:0':loc})
            start_epoch = checkpoint['epoch']
            # model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    extractor = model.encoder_q
    
    Classifi = Classifier(5)
    pretrain_dataset = ISICPretrainDatasetfoTrain(root='./ISIC/data/ISIC/', task_num=1, n_way=3, k_shot=k_shot, k_query=k_query)
    with torch.no_grad():
        for x, y in pretrain_dataset:
            x.cuda()
            output = extractor(x.squeeze())
            output_stack = output
            target_stack = y.squeeze()

    new_weight = torch.zeros(3, 128)
    for i in range(3):
        tmp = output_stack[target_stack == i].mean(0)
        new_weight[i] = tmp / tmp.norm(p=2)
    weight = new_weight.cuda()
    Classifi.layer[1].weight.data= weight
    del model
    return Classifi
    
def generate_npy():
    path = './ISIC/data/aptos/train.csv'
    img_path = './ISIC/data/aptos/images/'
    data = pd.read_csv(path)
    print(data[:20])
    d = os.listdir(img_path)
    
    data_dic = dict()
    for index, row in data.iterrows():
    #     print(list(row[1:][row[1:].values==1].index)[0])
    #     break
        img_name = row['id_code']
        label = row['diagnosis']
        # print(img_name, label)
        if label in data_dic.keys():
            data_dic[label].append(img_name)
        else:
            data_dic[label] = [img_name]

    x = []
    for label, imgs in tqdm(data_dic.items()):  # labels info deserted , each label contains 20imgs
        print(np.array(imgs).shape)
        x.append(np.array(imgs))

    # as different class may have different number of imgs
    # x = np.array(x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]

    # each character contains 20 imgs
    for i, a in enumerate(x):
        print('data shape:', a.shape)  # (1623, 20, 1, 28, 28)
        tmp = np.array(a)
        np.save(os.path.join('./ISIC/data/aptos/', f'class{i}.npy'), tmp)
        print('save class', i, 'to', os.path.join('./ISIC/data/aptos/', f'class{i}.npy'))

def count():
    seed = 111
    epoch = 190
    k_shot = 5
    path = f'./ISIC/data/ISIC/hardlabeldis2_seed{seed}_nt0{epoch}_{k_shot}.npy'
    hard_label = np.load(path)
    print(Counter(hard_label))
    print(Counter(hard_label[0:6705]))
    print(Counter(hard_label[6705:6705+1113]))
    print(Counter(hard_label[6705+1113:6705+1113+1099]))
    print(Counter(hard_label[6705+1113+1099:]))

if __name__ == '__main__':
    # calFeatureMean()
    # parse_acc()
    # initial_classifier()
    # generate_npy()
    count()