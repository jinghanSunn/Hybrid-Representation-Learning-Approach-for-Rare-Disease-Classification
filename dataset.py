# 2020.1.14 add line 21 transforms.Resize((224, 224))
import os

import torch
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import moco.loader
import moco.builder


class ISICDataset(Dataset):

    def __init__(self, root):
        """
        :param root:
        """
        self.path = os.path.join(root, 'ISIC2018_Task3_Training_Input/')
        self.transform = transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.)),
                                                transforms.RandomApply([
                                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0) 
                                                ], p=0.8),
                                                transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                 ])

        t_class = [0, 1, 2, 5] # largest amount of cases | (6750, 1113, 1099, 514)

        x_data, class_len = load_data(root, t_class)

        self.x_data = x_data

        print(class_len)

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        img = openImage(self.x_data[idx], self.path)
        q = self.transform(img)
        k = self.transform(img)
        return [q, k]



class ISICUnlabelDataset(Dataset):

    def __init__(self, root):
        """
        :param root:
        """
        self.path = os.path.join(root, 'ISIC2018_Task3_Training_Input/')
        self.transform = transforms.Compose([   
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            ])

        t_class = [0, 1, 2, 5] # largest amount of cases | (6750, 1113, 1099, 514)

        x_data, class_len = load_data(root, t_class) 

        self.x_data = x_data

        print(class_len)

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        img = openImage(self.x_data[idx], self.path)
        q = self.transform(img)
        return q
  


class ISICPsuedoDataset(Dataset):

    def __init__(self, root, label, filters=None):
        """
        :param root:
        """
        self.path = os.path.join(root, 'ISIC2018_Task3_Training_Input/')
        self.transform_ori = transforms.Compose([   
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                 ])
        self.transform = transforms.Compose([   
                                                transforms.Resize((224, 224)),
                                                transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.)),
                                                transforms.RandomApply([
                                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0)
                                                ], p=0.8),
                                                transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                 ])

        t_class = [0, 1, 2, 5] # largest amount of cases | (6750, 1113, 1099, 514)

        x_data, class_len = load_data(root, t_class)
        
        self.x_data = x_data # (9431,)
        self.label = label

        print(class_len)

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        img = openImage(self.x_data[idx], self.path)
        o = self.transform_ori(img)
        q = self.transform(img)
        k = self.transform(img)
        y = self.label[idx]
        return [o, q, k, y]


class ISICPretrainDataset(Dataset):

    def __init__(self, root, n_way, k_shot, k_query, aug_num):
        """
        :param transform:
        :param task_num: all training task num
        :param batchsz:
        :param n_way:
        :param k_shot:
        :param k_qry:
        """
        self.path = os.path.join(root, 'ISIC2018_Task3_Training_Input/')
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        self.augment_num = aug_num
    

        self.transform_spt = transforms.Compose([  
                                                transforms.Resize((224, 224)),
                                                transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.)),
                                                transforms.RandomApply([
                                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0) 
                                                ], p=0.8),
                                                transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                 ])
        self.transform_qry = transforms.Compose([
                                                 transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 ])
             
        t_class = [3, 4, 6] # test class

        x_data, class_len = load_data(root, t_class)
        
        print("ISICPretrainDataset", class_len)

        self.task = []
        selected_cls = [0, 1, 2] 
        x_spt, y_spt, x_qry, y_qry = [], [], [], []
        print("selected_cls", selected_cls)
        for j, cur_class in enumerate(selected_cls):

            selected_img = np.random.choice(class_len[j], k_shot + k_query, False) 

            x_spt.append(x_data[cur_class][selected_img[:k_shot]])
            x_qry.append(x_data[cur_class][selected_img[k_shot:]])
            y_spt.append([j for _ in range(k_shot)])
            y_qry.append([j for _ in range(k_query)])

        # shuffle inside a batch
        perm = np.random.permutation(self.n_way * self.k_shot)
        self.x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot)[perm]
        self.y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]

        perm = np.random.permutation(self.n_way * self.k_query)
        self.x_qry = np.array(x_qry).reshape(self.n_way * self.k_query)[perm]
        self.y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

    def __len__(self):
        return int(1)
    
    def __getitem__(self, index):
        i = 0
        for _ in range(self.augment_num):
            for j in range(len(self.x_spt)):
                img = openImage(self.x_spt[j], self.path)
                if i == 0:
                    x = np.expand_dims(self.transform_spt(img), axis=0)
                    y = np.expand_dims(self.y_spt[j], axis=0)
                    i = 1
                else:
                    x = np.concatenate((x, np.expand_dims(self.transform_spt(img),axis=0)), axis=0)
                    y = np.concatenate((y, np.expand_dims(self.y_spt[j],axis=0)), axis=0)

        for k in range(len(self.x_qry)):
            img = openImage(self.x_qry[k], self.path)
            if k==0:
                x_qry = np.expand_dims(self.transform_qry(img), axis=0)
                y_qry = np.expand_dims(self.y_qry[k], axis=0)
            else:
                x_qry = np.concatenate((x_qry, np.expand_dims(self.transform_qry(img),axis=0)), axis=0)
                y_qry = np.concatenate((y_qry, np.expand_dims(self.y_qry[k],axis=0)), axis=0)

        return x, y, x_qry, y_qry

    def get_pretrain_task(self):
        task = {"x_spt": self.x_spt, "y_spt": self.y_spt, "x_qry": self.x_qry, "y_qry": self.y_qry}
        return task

class ISICTestODataset(Dataset):

    def __init__(self, root, k_shot):
        """
        :param root:
        """
        self.k_shot = k_shot
        self.path = root + 'ISIC2018_Task3_Training_Input/'

        self.transform  = transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            ])
        
        t_class = [3, 4, 6] # test class

        pretrain_task = np.load(os.path.join(root,f'pretrain_task_shot{self.k_shot}.npy'), allow_pickle=True)
        pretrain_task = pretrain_task.item()
        pretrain_spt = pretrain_task['x_spt']

        class_len = []
        self.x = []
        self.label = []
        for i, c in enumerate(t_class):
            tmp = np.load(os.path.join(root, f'class{c}.npy'))
            tmp = [x for x in tmp if x not in pretrain_spt]
            self.x.extend(np.array(tmp))
            self.label.extend(i for _ in range(len(np.array(tmp))))
            class_len.append(len(np.array(tmp)))
        
        print(class_len)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x_qry_img = self.transform(openImage(self.x[idx], self.path)).float()
        y_qry = self.label[idx]
        return x_qry_img, y_qry


def load_data(root, t_class):
    class_len = []
    x_data = []
    for c in t_class:
        tmp = np.load(os.path.join(root, f'class{c}.npy'))
        x_data.extend(tmp)
        class_len.append(len(tmp))
    x_data = np.array(x_data) 
    return x_data, class_len

def openImage(x, path):
    return Image.open(path + x + '.jpg').convert('RGB')

if __name__ == '__main__':
    pass