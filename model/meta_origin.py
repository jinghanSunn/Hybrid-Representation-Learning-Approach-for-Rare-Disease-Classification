import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
from torch.optim.lr_scheduler import StepLR
import  numpy as np

from    model.learner import Learner
from    copy import deepcopy
from itertools import chain

from sklearn.metrics import roc_auc_score



class Meta_origin(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, model):
        """

        :param args:
        """
        super(Meta_origin, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_shot
        self.k_qry = args.k_query
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.device = torch.device('cuda')


        self.net = model
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        # self.lr_scheduler = StepLR(self.meta_optim, step_size=150, gamma=0.1)


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, task):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """

        task_num = len(task[0])
        setsz, c_, h, w = task[0][0].size()
        querysz = task[3][0].size(0)
        
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i [0,1,2,3,4,5,6] 0指初始值
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):  # 一个batch有b个tasks
            x_spt = task[0][i].to(self.device)
            y_spt = task[1][i].to(self.device)
            x_qry = task[2][i].to(self.device)
            y_qry = task[3][i].to(self.device)
            # print("y")
            # print(y_spt)
            # print(y_qry)
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt, vars=None, bn_training=True) # logits:5x5
            # print(logits)
            # print(y_spt)
            loss = F.cross_entropy(logits, y_spt)
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry, self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry)
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[0] = corrects[0] + correct # 初始值预测正确的个数？

            if self.update_step == 1:
                # [setsz, nway]
                logits_q = self.net(x_qry, fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry)
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[1] = corrects[1] + correct
            
            else:
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry, fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry)
                    losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry).sum().item()
                    corrects[1] = corrects[1] + correct


                for k in range(1, self.update_step): # support set 论文中update_step应该设的就是1
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt, fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, y_spt)
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry, fast_weights, bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    # print(logits_q)
                    # print(y_qry)
                    loss_q = F.cross_entropy(logits_q, y_qry)
                    losses_q[k + 1] += loss_q
                    # losses_q[k + 1] += (loss_q)**5 * torch.log(torch.max(1e-4, 1-loss_q))

                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num # 所有task都update完毕，用最后一次update step得到的loss更新参数

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()
        # self.lr_scheduler.step()

        accs = np.array(corrects) / (querysz * task_num)


        return accs, loss_q


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct
            

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                logits_q_sm = F.softmax(logits_q, dim=1)
                pred_q = logits_q_sm.argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        accs = np.array(corrects) / querysz
        AUC = roc_auc_score(np.eye(2)[y_qry.cpu().numpy()], logits_q_sm.cpu().numpy())

        return accs, AUC



def main():
    pass


if __name__ == '__main__':
    main()
