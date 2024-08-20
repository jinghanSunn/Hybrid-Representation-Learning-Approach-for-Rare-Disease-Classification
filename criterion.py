import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

class DistillKL_weight(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T, weight=None):
        super(DistillKL_weight, self).__init__()
        self.T = T
        # self.weight = weight

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        # print(self.weight)
        # print(F.softmax(self.weight,dim=0))
        # print(F.kl_div(p_s, p_t, reduction='none'))
        # print(torch.sum(F.kl_div(p_s, p_t, reduction='none'), dim=-1))
        # print(F.kl_div(p_s, p_t, reduction='sum'))
        # print(self.weight.shape)
        # print(self.weight * F.kl_div(p_s, p_t, reduction='none'))
        weight = torch.ones((y_s.shape[0]))
        weight[y_t.argmax(dim=1)==0] = 1.5
        weight = weight.unsqueeze(1).cuda()
        # print(weight)
        loss = torch.sum(weight * F.kl_div(p_s, p_t, reduction='none')) * (self.T**2) / y_s.shape[0]
        return loss

def my_cross_entropy(input, target, weight, reduction="mean"):
	# input.shape: torch.size([-1, class])
	# target.shape: torch.size([-1])
	# reduction = "mean" or "sum"
	# input是模型输出的结果，与target求loss
	# target的长度和input第一维的长度一致
	# target的元素值为目标class
	# reduction默认为mean，即对loss求均值
	# 还有另一种为sum，对loss求和

	# 这里对input所有元素求exp
    exp = torch.exp(input)
    # 根据target的索引，在exp第一维取出元素值，这是softmax的分子
    tmp1 = exp.gather(1, target.unsqueeze(-1)).squeeze()
    # 在exp第一维求和，这是softmax的分母
    tmp2 = exp.sum(1)
	# softmax公式：ei / sum(ej)
    softmax = tmp1 / tmp2
    # cross-entropy公式： -yi * log(pi)
    # 因为target的yi为1，其余为0，所以在tmp1直接把目标拿出来，
    # 公式中的pi就是softmax的结果
    # print(softmax.shape)
    log = -weight*torch.log(softmax)
    # 官方实现中，reduction有mean/sum及none
    # 只是对交叉熵后处理的差别
    if reduction == "mean": return log.mean()
    else: return log.sum()


class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1, 
        base_loss_factor=1.0, additional_diversity_factor=1, reweight_factor=0.1):
        super().__init__()
        # self.base_loss = F.cross_entropy
        # self.base_loss_factor = base_loss_factor
        # if not reweight:
        #     self.reweight_epoch = -1
        # else:
        #     self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        # if cls_num_list is None:
        #     # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

        #     self.m_list = None
        #     self.per_cls_weights_enabled = None
        #     self.per_cls_weights_enabled_diversity = None
        # else:
            # We will use LDAM loss if we provide cls_num_list.

        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
        self.m_list = m_list
        self.s = s
        assert s > 0
        
        # if reweight_epoch != -1:
        idx = 1 # condition could be put in order to set idx
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list) # a list of number in [0,1]
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        # else:
        #     self.per_cls_weights_enabled = None

        cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
        C = len(cls_num_list)
        per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor # reweight_factor:gamma? maybe change

        # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
        # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
        per_cls_weights = per_cls_weights / np.max(per_cls_weights)

        assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
        # save diversity per_cls_weights
        self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor
        self.per_cls_weights_base = self.per_cls_weights_enabled
        self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    # def _hook_before_epoch(self, epoch):
    #     if self.reweight_epoch != -1:
    #         self.epoch = epoch

    #         if epoch > self.reweight_epoch:
    #             self.per_cls_weights_base = self.per_cls_weights_enabled
    #             self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
    #         else:
    #             self.per_cls_weights_base = None
    #             self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
            
        # base_diversity_temperature = self.base_diversity_temperature

        # if self.per_cls_weights_diversity is not None:
        diversity_temperature = 1 * self.per_cls_weights_diversity.view((1, -1))
        # print(diversity_temperature) 
        # 1shot: [1, 0.9840, 0.9997]  
        # 3shot: [1, 0.9891, 0.9854]  
        # 5shot: [0.9783, 1, 0.9578]  
        temperature_mean = diversity_temperature.mean().item()
        # else:
        #     diversity_temperature = 1
        #     temperature_mean = 1
        
        output_dist = F.log_softmax(output_logits / diversity_temperature, dim=1)
        with torch.no_grad():
            # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
            mean_output_dist = F.softmax(target / diversity_temperature, dim=1)
        
        loss = 1 * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='sum')/target.shape[0]
        
        return loss