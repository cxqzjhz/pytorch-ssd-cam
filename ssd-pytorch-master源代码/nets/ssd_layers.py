from __future__ import division
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
from torch.autograd import Variable
from math import sqrt as sqrt
from itertools import product as product
import numpy as np
from utils.box_utils import decode, nms
from utils.config import Config

# class Detect(Function)用户只实现了forward方法而没有实现backward方法,
# 是因为该子类只用于推理,而不用于训练,故只需要前向传递不需要反向传播
class Detect(Function):


    """     def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        # Detect(num_classes, 0, 200, 0.01, 0.45)  # num_classes = 3
        self.num_classes = num_classes  # 3
        self.background_label = bkg_label  # 0
        self.top_k = top_k  # 200
        self.nms_thresh = nms_thresh  # 0.45
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh  # 0.01
        self.variance = Config['variance']  # [0.1, 0.2] 
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        # Detect(num_classes, 0, 200, 0.01, 0.45)  # num_classes = 3
        Detect.num_classes = num_classes  # 3  # 注意:原本是self.xxx = xxx 现在改为类对象的属性,而非实例属性
        Detect.background_label = bkg_label  # 0
        Detect.top_k = top_k  # 200
        Detect.nms_thresh = nms_thresh  # 0.45
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        Detect.conf_thresh = conf_thresh  # 0.01
        Detect.variance = Config['variance']  # [0.1, 0.2]
    # class Exp(Function):
    # @staticmethod
    # def forward(ctx, i):
    # output = Exp.apply(input)
    @staticmethod
    def forward(ctx, loc_data, conf_data, prior_data):
        # # loc_data preds torch.Size([1, 8732, 4])
        # # conf_data  # torch.Size([1, 8732, 3]) 
        # # prior_data torch.Size([8732, 4])
        loc_data = loc_data.cpu()
        conf_data = conf_data.cpu()
        num = loc_data.size(0)  # batch size 1
        num_priors = prior_data.size(0)  # 8732
        output = torch.zeros(num, Detect.num_classes, Detect.top_k, 5)  # torch.Size([1, 3, 200, 5])
        conf_preds = conf_data.view(num, num_priors,
                                    Detect.num_classes).transpose(2, 1)  # torch.Size([1, 3, 8732])
        # 对每一张图片进行处理
        for i in range(num):
            # 对先验框解码获得预测框
            decoded_boxes = decode(loc_data[i], prior_data, Detect.variance)  # torch.Size([8732, 4])
            conf_scores = conf_preds[i].clone()  # torch.Size([3, 8732])

            for cl in range(1, Detect.num_classes):  # 遍历1到2,因为0代表背景
                # 对每一类进行非极大抑制
                c_mask = conf_scores[cl].gt(Detect.conf_thresh)  # 获取正样本的索引  torch.Size([8732])
                scores = conf_scores[cl][c_mask]  # 获取所有正样本的置信度分数  torch.Size([11])
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)  # torch.Size([8732, 4])
                boxes = decoded_boxes[l_mask].view(-1, 4)  # torch.Size([11, 4]) 获取所有正样本的边框
                # 进行非极大抑制
                ids, count = nms(boxes, scores, Detect.nms_thresh, Detect.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)  # 这几行代码注释掉之后程序仍然能够正确运行
        _, idx = flt[:, :, 0].sort(1, descending=True)  # 这几行代码注释掉之后程序仍然能够正确运行
        _, rank = idx.sort(1)  # 这几行代码注释掉之后程序仍然能够正确运行
        flt[(rank < Detect.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)  # 这几行代码注释掉之后程序仍然能够正确运行  
        # 注意这里的操作并不会影响output,因为flt[mask].fill_(0)不会影响output
        return output  # torch.Size([1, 3, 200, 5])  1置信度+4位置信息


'''
    def forward(self, loc_data, conf_data, prior_data):
        # # loc_data preds torch.Size([1, 8732, 4])
        # # conf_data  # torch.Size([1, 8732, 3]) 
        # # prior_data torch.Size([8732, 4])
        loc_data = loc_data.cpu()
        conf_data = conf_data.cpu()
        num = loc_data.size(0)  # batch size 1
        num_priors = prior_data.size(0)  # 8732
        output = torch.zeros(num, self.num_classes, self.top_k, 5)  # torch.Size([1, 3, 200, 5])
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)  # torch.Size([1, 3, 8732])
        # 对每一张图片进行处理
        for i in range(num):
            # 对先验框解码获得预测框
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)  # torch.Size([8732, 4])
            conf_scores = conf_preds[i].clone()  # torch.Size([3, 8732])

            for cl in range(1, self.num_classes):  # 遍历1到2,因为0代表背景
                # 对每一类进行非极大抑制
                c_mask = conf_scores[cl].gt(self.conf_thresh)  # 获取正样本的索引  torch.Size([8732])
                scores = conf_scores[cl][c_mask]  # 获取所有正样本的置信度分数  torch.Size([11])
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)  # torch.Size([8732, 4])
                boxes = decoded_boxes[l_mask].view(-1, 4)  # torch.Size([11, 4]) 获取所有正样本的边框
                # 进行非极大抑制
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)  # 这几行代码注释掉之后程序仍然能够正确运行
        _, idx = flt[:, :, 0].sort(1, descending=True)  # 这几行代码注释掉之后程序仍然能够正确运行
        _, rank = idx.sort(1)  # 这几行代码注释掉之后程序仍然能够正确运行
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)  # 这几行代码注释掉之后程序仍然能够正确运行  
        # 注意这里的操作并不会影响output,因为flt[mask].fill_(0)不会影响output
        return output  # torch.Size([1, 3, 200, 5])  1置信度+4位置信息
'''

# Config = {
#     'num_classes': 3, # 'num_classes': 21,
#     'feature_maps': [38, 19, 10, 5, 3, 1],
#     'min_dim': 300,
#     'steps': [8, 16, 32, 64, 100, 300],
#     'min_sizes': [30, 60, 111, 162, 213, 264],
#     'max_sizes': [60, 111, 162, 213, 264, 315],
#     'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
#     'variance': [0.1, 0.2],
#     'clip': True,
#     'name': 'VOC',
# }

class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']  # 300
        self.num_priors = len(cfg['aspect_ratios'])  # 6
        self.variance = cfg['variance'] or [0.1]  # [0.1, 0.2]
        self.feature_maps = cfg['feature_maps']  # [38, 19, 10, 5, 3, 1]
        self.min_sizes = cfg['min_sizes']  # [30, 60, 111, 162, 213, 264]
        self.max_sizes = cfg['max_sizes']  # [60, 111, 162, 213, 264, 315]
        self.steps = cfg['steps']  # [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = cfg['aspect_ratios']  # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.clip = cfg['clip']  # True
        self.version = cfg['name']  # VOC
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):  # [38, 19, 10, 5, 3, 1]
            x,y = np.meshgrid(np.arange(f),np.arange(f))  # 笛卡尔坐标形式 38 x 38
            x = x.reshape(-1)
            y = y.reshape(-1)
            for i, j in zip(y,x):
                f_k = self.image_size / self.steps[k]  # 300 / [8,16,32,64,100,300] 计算每个网格的像素宽度
                # 300 / [8,16,32,64,100,300] 向上取整对应 # [38, 19, 10, 5, 3, 1]
                # 计算网格的中心
                cx = (j + 0.5) / f_k  # 中心点横坐标位置相对于特征图整体的比例
                cy = (i + 0.5) / f_k  # 中心点纵坐标位置相对于特征图整体的比例

                # 求短边  # 占整个图片的比例
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 求长边  # 占整个图片的比例
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 获得长方形
                for ar in self.aspect_ratios[k]:  # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]  # 获得不同宽高比的先验框
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]  # 获得不同宽高比的先验框
        # 获得所有的先验框
        # mean的数据 # 中心点的横坐标位置 # 宽度和高度 (占整个图片尺寸的比例)
        output = torch.Tensor(mean).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)
        return output  # torch.Size([8732, 4])

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))  # 长度是512的权重 torch.Size([512]) 
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps  # torch.Size([4, 1, 38, 38])
        #x /= norm
        x = torch.div(x,norm)  # torch.Size([4, 512, 38, 38])
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out  # torch.Size([4, 512, 38, 38])
