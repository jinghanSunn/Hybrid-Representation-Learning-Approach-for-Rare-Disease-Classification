# 2020.1.14 add line 21 transforms.Resize((224, 224))
from collections import Counter
import os

import torch
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import moco.loader
import moco.builder
from sklearn.model_selection import train_test_split

seed = 222 
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
                                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0)  # not strengthened # previous:0.4, 0.4, 0.4, 0.1
                                                ], p=0.8),
                                                # transforms.RandomGrayscale(p=0.2),
                                                transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                 ])
        

        x = []
        for i in range(7):
            x.append(np.load(os.path.join(root, f'class{i}.npy')))

        t_class = [0, 1, 2, 5] # largest amount of cases | (6750, 1113, 1099, 514)

        class_len = []
        x_data = []
        for c in t_class:
            x_data.extend(x[c])
            class_len.append(len(x[c]))

        self.x_data = np.array(x_data) # (9431,)

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
        

        x = []
        for i in range(7):
            x.append(np.load(os.path.join(root, f'class{i}.npy')))

        t_class = [0, 1, 2, 5] # largest amount of cases | (6750, 1113, 1099, 514)
        # else:
        # t_class = [3, 4, 6] # test class

        class_len = []
        x_data = []
        # label = []
        for c in t_class:
            x_data.extend(x[c])
            class_len.append(len(x[c]))
            # label.extend(c for _ in range(len(x[c])))
        self.x_data = np.array(x_data) # (9431,)
        # self.label = np.array(label)
        print(class_len)

    def __len__(self):
        return len(self.x_data)
        # return 10
    
    def __getitem__(self, idx):
        img = openImage(self.x_data[idx], self.path)
        q = self.transform(img)
        return q
        # return q, self.label[idx]

class ISIClabelDataset(Dataset):

    def __init__(self, root):
        """
        :param root:
        """
        self.path = os.path.join(root, 'ISIC2018_Task3_Training_Input/')
        self.transform = transforms.Compose([   
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                 ])
        

        x = []
        for i in range(7):
            x.append(np.load(os.path.join(root, f'class{i}.npy')))

        t_class = [0, 1, 2, 3, 4, 5, 6] # largest amount of cases | (6750, 1113, 1099, 514)
        # else:
        # t_class = [3, 4, 6] # test class

        class_len = []
        x_data = []
        label = []
        for c in t_class:
            if c in [0,1,2,5]:
                tmp = x[c][:50]
            else:
                tmp = x[c]
            x_data.extend(tmp)
            class_len.append(len(tmp))
            label.extend(c for _ in range(len(tmp)))
        label = np.array(label)
        label[(label==0) | (label==1) | (label==2) | (label==5)] = 0
        label[(label==3) | (label==4) | (label==6)] = 1
        self.x_data = np.array(x_data) # (9431,)
        self.label = np.array(label)
        print("label",np.unique(self.label))
        print(np.sum(label==0), np.sum(label==1))
        print(class_len)
        

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        img = openImage(self.x_data[idx], self.path)
        q = self.transform(img)
        # return q
        return q, self.label[idx]

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
                                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0)  # not strengthened # previous:0.4, 0.4, 0.4, 0.1
                                                ], p=0.8),
                                                # transforms.RandomGrayscale(p=0.2),
                                                transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                 ])
        

        x = []
        for i in range(7):
            x.append(np.load(os.path.join(root, f'class{i}.npy')))

        t_class = [0, 1, 2, 5] # largest amount of cases | (6750, 1113, 1099, 514)
        # else:
        #     t_class = [3, 4, 6] # test class

        class_len = []
        x_data = []
        for c in t_class:
            x_data.extend(x[c])
            class_len.append(len(x[c]))
        x_data = np.array(x_data) 
        # if filters=="lc": # least confidence, just count
        #     # print(np.max(label,axis=-1)[:100])
        #     index = np.max(label,axis=-1)>0.7
        #     print("pesudo label length:", np.sum(index))
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


class ISICDistillDataset(Dataset):

    def __init__(self, root, t_class = [0, 1, 2, 5]):
        """
        :param root:
        """
        self.path = os.path.join(root, 'ISIC2018_Task3_Training_Input/')
        self.transform = transforms.Compose([   transforms.Resize((224, 224)),
                                                transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.)),
                                                # transforms.RandomGrayscale(p=0.2),
                                                transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                                                transforms.RandomVerticalFlip(),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                 ])
        

        x = []
        for i in range(7):
            x.append(np.load(os.path.join(root, f'class{i}.npy')))

         # largest amount of cases | (6750, 1113, 1099, 514)
        # else:
        #     t_class = [3, 4, 6] # test class

        class_len = []
        x_data = []
        x_label = []
        for i, c in enumerate(t_class):
            x_data.extend(x[c])
            x_label.extend(i for _ in range(len(x[c])))
            class_len.append(len(x[c]))

        self.x_data = np.array(x_data) # (9431,)
        self.x_label = np.array(x_label)
        print(self.x_data.shape)
        print(self.x_label.shape)

        print(class_len)

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.transform(openImage(self.x_data[idx], self.path))
        y = self.x_label[idx]
        return [x, y]

class ISICPretrainDatasetfoTrain(Dataset):

    def __init__(self, root, task_num, n_way, k_shot, k_query):
        """
        :param root:
        :param task_num: all training/test task num
        :param batchsz:
        :param n_way:
        :param k_shot:
        :param k_qry:
        """
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        self.path = root + 'ISIC2018_Task3_Training_Input/'


        x = []
        for i in range(7):
            x.append(np.load(os.path.join(root, f'class{i}.npy')))
        
        t_class = [3, 4, 6] # test class

        class_len = []
        x_data = []
        for c in t_class:
            x_data.append(x[c])
            class_len.append(len(x[c]))
        
        print(class_len)

        self.transform_spt = transforms.Compose([   
                                                transforms.Resize((224, 224)),
                                            
                                                transforms.ToTensor(),
                                                 ])
        self.transform_qry = transforms.Compose([
                                                 transforms.Resize((224, 224)),
                                                #  transforms.CenterCrop((90, 120)),
                                                 transforms.ToTensor(),
                                                 ])


        # pretrain_task = np.load(os.path.join(root,f'pretrain_task_shot{self.k_shot}.npy'), allow_pickle=True)
        pretrain_task = np.load(os.path.join(root,f'pretrain_task_seed{seed}_shot{self.k_shot}.npy'), allow_pickle=True)
        print("load from ", os.path.join(root,f'pretrain_task_seed{seed}_shot{self.k_shot}.npy'))
        pretrain_task = pretrain_task.item()
        self.x_spt = pretrain_task['x_spt']
        self.y_spt = pretrain_task['y_spt']
        self.x_qry = pretrain_task['x_qry']
        self.y_qry = pretrain_task['y_qry']


    def __len__(self):
        return int(1)
    
    def __getitem__(self, index):
        for j in range(len(self.x_spt)):
            img = openImage(self.x_spt[j], self.path)
            if j == 0:
                x = np.expand_dims(self.transform_spt(img), axis=0)
                y = np.expand_dims(self.y_spt[j], axis=0)
            else:
                x = np.concatenate((x, np.expand_dims(self.transform_spt(img),axis=0)), axis=0)
                y = np.concatenate((y, np.expand_dims(self.y_spt[j],axis=0)), axis=0)
        return x, y

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
                                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0)  # not strengthened # previous:0.4, 0.4, 0.4, 0.1
                                                ], p=0.8),
                                                # transforms.RandomGrayscale(p=0.2),
                                                transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                 ])
        self.transform_qry = transforms.Compose([
                                                 transforms.Resize((224, 224)),
                                                #  transforms.CenterCrop((90, 120)),
                                                 transforms.ToTensor(),
                                                 ])
        
        x = []
        for i in range(7):
            x.append(np.load(os.path.join(root, f'class{i}.npy')))
        
        t_class = [3, 4, 6] # test class

        class_len = []
        x_data = []
        for c in t_class:
            x_data.append(x[c])
            class_len.append(len(x[c]))
        
        print("ISICPretrainDataset", class_len)
        print("transform")
        print(self.transform_spt)
        print(self.transform_qry)

        self.task = []
        selected_cls = [0, 1, 2] # ???
        x_spt, y_spt, x_qry, y_qry = [], [], [], []
        print("selected_cls", selected_cls)
        for j, cur_class in enumerate(selected_cls):

            selected_img = np.random.choice(class_len[j], k_shot + k_query, False) # 在每一类中随机选择k_shot+k_query个图片

            x_spt.append(x_data[cur_class][selected_img[:k_shot]])
            x_qry.append(x_data[cur_class][selected_img[k_shot:]])
            y_spt.append([j for _ in range(k_shot)]) # label与真实label无关，每个episode的类别都只有n_way个， 当前选中的图片类别为j
            y_qry.append([j for _ in range(k_query)])

        # shuffle inside a batch
        perm = np.random.permutation(self.n_way * self.k_shot)
        self.x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot)[perm]
        self.y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
        # y_spt = np.eye(n_way)[y_spt]

        perm = np.random.permutation(self.n_way * self.k_query)
        self.x_qry = np.array(x_qry).reshape(self.n_way * self.k_query)[perm]
        self.y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

    def __len__(self):
        return int(1)
    
    def __getitem__(self, index):
        # print(self.x_spt[index])
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
        print(self.x_spt)
        task = {"x_spt": self.x_spt, "y_spt": self.y_spt, "x_qry": self.x_qry, "y_qry": self.y_qry}
        return task


class ISICTestDataset(Dataset):

    def __init__(self, root, task_num, n_way, k_shot, k_query, selected_cls=None, test='normal'):
        """
        :param root:
        :param task_num: all training/test task num
        :param batchsz:
        :param n_way:
        :param k_shot:
        :param k_qry:
        """
        self.n_cls = 3
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        self.path = root + 'ISIC2018_Task3_Training_Input/'
        self.test = test

        transform_test = transforms.Compose([
                                                # lambda x: Image.open(path + x + '.jpg').convert('RGB'),
                                                 transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 ])
        self.transform = transform_test

        x = []
        for i in range(7):
            x.append(np.load(os.path.join(root, f'class{i}.npy')))
        
        t_class = [3, 4, 6] # test class

        # class_len = []
        # self.x_data = []
        # for c in t_class:
        #     self.x_data.append(x[c])
        #     class_len.append(len(x[c]))

        # exclude the data used in pretrain
        # pretrain_task = np.load(os.path.join(root, f'pretrain_task_shot{self.k_shot}.npy'), allow_pickle=True)
        pretrain_task = np.load(os.path.join(root,f'pretrain_task_seed{seed}_shot{self.k_shot}.npy'), allow_pickle=True)
        pretrain_task = pretrain_task.item()
        pretrain_spt = pretrain_task['x_spt']
        print(pretrain_spt)
        class_len = []
        self.x_data = []
        for c in t_class:
            tmp = x[c]
            tmp = [x for x in tmp if x not in pretrain_spt]
            self.x_data.append(np.array(tmp))
            class_len.append(len(tmp))
        
        # self.x_data = np.array(self.x_data)
        print(class_len)
        print("selected_cls", selected_cls)

        self.task = []
        if selected_cls == None:
            for _ in range(task_num):
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(self.n_cls, self.n_way, False) 
                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(class_len[cur_class], self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    x_spt.append(self.x_data[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(self.x_data[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)]) # label与真实label无关，每个episode的类别都只有n_way个， 当前选中的图片类别为j
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                # y_spt = np.eye(n_way)[y_spt]

                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]
                # y_qry = np.eye(n_way)[y_qry]

                self.task.append([x_spt, y_spt, x_qry, y_qry])
        else:
            if test == 'hard':
                print("test on selected class:", selected_cls)
                # ratio = [15*]
                print("hard test")
                for _ in range(task_num):
                    x_spt, y_spt, x_qry, y_qry = [], [], [], []
                    self.query_num = 0
                    for j, cur_class in enumerate(selected_cls):

                        # selected_img = np.random.choice(class_len[cur_class], self.k_shot + self.k_query, False)
                        query_k = int(self.k_query*self.n_way*class_len[cur_class]/np.sum(class_len))
                        self.query_num += query_k
                        selected_img = np.random.choice(class_len[cur_class], self.k_shot + query_k, False)

                        # meta-training and meta-test
                        x_spt.append(self.x_data[cur_class][selected_img[:self.k_shot]])
                        x_qry.extend(self.x_data[cur_class][selected_img[self.k_shot:]])
                        y_spt.append([j for _ in range(self.k_shot)]) # label与真实label无关，每个episode的类别都只有n_way个， 当前选中的图片类别为j
                        # y_qry.extend([j for _ in range(self.k_query)])
                        y_qry.extend([j for _ in range(query_k)])

                    # shuffle inside a batch
                    perm = np.random.permutation(self.n_way * self.k_shot)
                    x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot)[perm]
                    y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
               
                    perm = np.random.permutation(self.query_num)
                    x_qry = np.array(x_qry).reshape(self.query_num)[perm]
                    y_qry = np.array(y_qry).reshape(self.query_num)[perm]
                    # y_qry = np.eye(n_way)[y_qry]

                    self.task.append([x_spt, y_spt, x_qry, y_qry])

            elif test == 'normal':
                # normal test
                print("test on selected class:", selected_cls)
                for _ in range(task_num):
                    x_spt, y_spt, x_qry, y_qry = [], [], [], []
                    for j, cur_class in enumerate(selected_cls):
                        selected_img = np.random.choice(class_len[cur_class], self.k_shot + self.k_query, False)
                       
                        # meta-training and meta-test
                        x_spt.append(self.x_data[cur_class][selected_img[:self.k_shot]])
                        x_qry.append(self.x_data[cur_class][selected_img[self.k_shot:]])
                        y_spt.append([j for _ in range(self.k_shot)])
                        y_qry.append([j for _ in range(self.k_query)])

                    # shuffle inside a batch
                    perm = np.random.permutation(self.n_way * self.k_shot)
                    x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot)[perm]
                    y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                    # y_spt = np.eye(n_way)[y_spt]

                    perm = np.random.permutation(self.n_way * self.k_query)
                    x_qry = np.array(x_qry).reshape(self.n_way * self.k_query)[perm]
                    y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]
                    # y_qry = np.eye(n_way)[y_qry]

                    self.task.append([x_spt, y_spt, x_qry, y_qry])
                self.query_num = self.n_way * self.k_query
            else:
                raise NotImplementedError

    def __len__(self):
        return len(self.task)
    
    def __getitem__(self, idx):
        #  take 5 way 1 shot as example: 5 * 1
        task = self.task[idx]
        x_spt, y_spt, x_qry, y_qry = task
        # x_spt_img, x_qry_img = torch.zeros((self.n_way * self.k_shot, 3, 224, 224)), torch.zeros((self.n_way * self.k_query, 3, 224, 224))
        x_spt_img, x_qry_img = torch.zeros((self.n_way * self.k_shot, 3, 224, 224)), torch.zeros((self.query_num, 3, 224, 224))
        
        for i in range(len(x_spt)):
            x_spt_img[i] = self.transform(openImage(x_spt[i], self.path)).float()

        for i in range(len(x_qry)):
            x_qry_img[i] = self.transform(openImage(x_qry[i], self.path)).float()
        
        x_spt_img = x_spt_img.reshape(self.n_way * self.k_shot, 3, 224, 224)
        # x_qry_img = x_qry_img.reshape(self.n_way * self.k_query, 3, 224, 224)
        x_qry_img = x_qry_img.reshape(self.query_num, 3, 224, 224)
        # print(x_spt_img.shape)
        return x_spt_img, y_spt, x_qry_img, y_qry

class ISICTestODataset(Dataset):

    def __init__(self, root, task_num, n_way, k_shot, k_query, selected_cls=None):
        """
        :param root:
        :param task_num: all training/test task num
        :param batchsz:
        :param n_way:
        :param k_shot:
        :param k_qry:
        """
        self.n_cls = 3
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        self.path = root + 'ISIC2018_Task3_Training_Input/'

        transform_test = transforms.Compose([
                                                # lambda x: Image.open(path + x + '.jpg').convert('RGB'),
                                                 transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 ])
        self.transform = transform_test

        x = []
        for i in range(7):
            x.append(np.load(os.path.join(root, f'class{i}.npy')))
        
        t_class = [3, 4, 6] # test class

        # pretrain_task = np.load(os.path.join(root, f'pretrain_task_shot{self.k_shot}.npy'), allow_pickle=True)
        pretrain_task = np.load(os.path.join(root,f'pretrain_task_seed{seed}_shot{self.k_shot}.npy'), allow_pickle=True)
        pretrain_task = pretrain_task.item()
        pretrain_spt = pretrain_task['x_spt']

        class_len = []
        self.x_data = []
        self.x = []
        self.label = []
        for i, c in enumerate(t_class):
            tmp = x[c]
            tmp = [x for x in tmp if x not in pretrain_spt]
            self.x_data.append(np.array(tmp))
            self.x.extend(np.array(tmp))
            self.label.extend(i for _ in range(len(np.array(tmp))))
            class_len.append(len(np.array(tmp)))
        
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.label, test_size=0.7, random_state=42)
        self.x = X_test
        self.label = y_test
        print(Counter(self.label))

        print(class_len)
        print("selected_cls", selected_cls)

        self.task = []
        if selected_cls == None:
            for _ in range(task_num):
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(self.n_cls, self.n_way, False) 
                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(class_len[cur_class], self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    x_spt.append(self.x_data[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(self.x_data[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)]) # label与真实label无关，每个episode的类别都只有n_way个， 当前选中的图片类别为j
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                # y_spt = np.eye(n_way)[y_spt]

                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]
                # y_qry = np.eye(n_way)[y_qry]

                self.task.append([x_spt, y_spt, x_qry, y_qry])
        else:
            # normal test
            print("test on selected class:", selected_cls)
            for _ in range(1):
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(class_len[cur_class], self.k_shot + self.k_query, False)
                    selected_img = np.random.choice(class_len[cur_class], self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    x_spt.append(self.x_data[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(self.x_data[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)]) # label与真实label无关，每个episode的类别都只有n_way个， 当前选中的图片类别为j
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                # y_spt = np.eye(n_way)[y_spt]

                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]
                # y_qry = np.eye(n_way)[y_qry]

                self.task.append([x_spt, y_spt, x_qry, y_qry])
            self.query_num = self.n_way * self.k_query

    def __len__(self):
        return len(self.x)
        # return 60
        # return 10
    
    def __getitem__(self, idx):
        #  take 5 way 1 shot as example: 5 * 1
        task = self.task[0]
        
        x_spt, y_spt, x_qry, y_qry = task
        # x_spt_img, x_qry_img = torch.zeros((self.n_way * self.k_shot, 3, 224, 224)), torch.zeros((self.n_way * self.k_query, 3, 224, 224))
        x_spt_img, x_qry_img = torch.zeros((self.n_way * self.k_shot, 3, 224, 224)), torch.zeros((self.n_way * self.k_shot, 3, 224, 224))
        
        for i in range(len(x_spt)):
            x_spt_img[i] = self.transform(openImage(x_spt[i], self.path)).float()
        
        x_spt_img = x_spt_img.reshape(self.n_way * self.k_shot, 3, 224, 224)
        x_qry_img = self.transform(openImage(self.x[idx], self.path)).float()
        y_qry = self.label[idx]
        return x_spt_img, y_spt, x_qry_img, y_qry



def openImage(x, path):
    return Image.open(path + x + '.jpg').convert('RGB')

if __name__ == '__main__':
    ISICDataset('/data3/jhsun/isic/')