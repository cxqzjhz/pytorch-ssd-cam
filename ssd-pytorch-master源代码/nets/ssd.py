import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.config import Config
from nets.ssd_layers import Detect
from nets.ssd_layers import L2Norm,PriorBox
from nets.vgg import vgg as add_vgg


class SSD(nn.Module):
    def __init__(self, phase, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = Config
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
            # self.priors = self.priorbox.forward()  # 这一行改成这样也能正常运行

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.relu_list4cxq = nn.ModuleList([torch.nn.ReLU(True) for i in range(8)])  # 自己修改后的方式
        self.feature_maps4cxq = None  # 用于grad cam
        self.scores4cxq = None  # 用于grad cam
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # 获得conv4_3的内容
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # 获得fc7的内容
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # 获得后面的内容
        for k, v in enumerate(self.extras):
            # x = F.relu(v(x), inplace=True)  # 原始实现方式
            x = self.relu_list4cxq[k](v(x))  # 修改后的方式  
            if k % 2 == 1:
                sources.append(x)


        self.feature_maps4cxq = sources  # 6张特征图
        # 添加回归层和分类层
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        self.scores4cxq = conf  # 用于保存各个类别的分数

        # 进行resize
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)  # torch.Size([4, 34928])
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)  # torch.Size([4, 26196])
        if self.phase == "test":
            # loc会resize到batch_size,num_anchors,4
            # conf会resize到batch_size,num_anchors,num_classes
            # output = self.detect(
            output = self.detect.apply(
                loc.view(loc.size(0), -1, 4),                   # loc preds torch.Size([4, 8732, 4])
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds # torch.Size([4, 8732, 3])
                self.priors              # torch.Size([8732, 4])
            )  # torch.Size([1, 3, 200, 5])  1置信度+4位置信息
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )  # torch.Size([4, 8732, 4]) torch.Size([4, 8732, 3]) torch.Size([8732, 4])
        return output


def add_extras(i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i

    # Block 6
    # 19,19,1024 -> 10,10,512
    layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

    # Block 7
    # 10,10,512 -> 5,5,256
    layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    # Block 8
    # 5,5,256 -> 3,3,256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    
    # Block 9
    # 3,3,256 -> 1,1,256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    return layers

mbox = [4, 6, 6, 6, 4, 4]

def get_ssd(phase,num_classes):  # get_ssd("train", 3)

    vgg, extra_layers = add_vgg(3), add_extras(1024)

    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 mbox[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        mbox[k] * num_classes, kernel_size=3, padding=1)]
                        
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                  * num_classes, kernel_size=3, padding=1)]

    SSD_MODEL = SSD(phase, vgg, extra_layers, (loc_layers, conf_layers), num_classes)
    return SSD_MODEL
