from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import cv2
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import os

MEANS = (104, 117, 123)
class SSDDataset(Dataset):
    def __init__(self, train_lines, image_size):
        super(SSDDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        """实时数据增强的随机预处理"""
        line = annotation_line.split()
        # print("cxq陈旭旗cxq文件路径:  ",line[0])
        # image = Image.open(line[0])
        myPath = line[0]
        if os.path.exists(myPath):
            # print(myPath)
            pass
        else:
            myPath = myPath[:-3] + "bmp"
            # print(myPath)

        image = Image.open(myPath)

        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # 调整图片大小
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)  # 对图片按照给定尺寸缩放,但不裁剪

        # 放置图片
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new(
            'RGB', (w, h),
            (
                np.random.randint(0, 255), 
                np.random.randint(0, 255), np.random.randint(0, 255)
            )
        )
        new_image.paste(image, (dx, dy))  # 原图缩放后粘贴到一张新的空图片上
        image = new_image

        # 是否翻转图片
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域变换
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)   
        # x 是<class 'numpy.ndarray'>  # (667, 1000, 3)
        # 0.0 359.34784   H
        # 0.0 0.9999999   S
        # 0.0 1.0         V
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1  # 感觉没有必要,注释后仍能正常运行##################################################
        x[..., 0][x[..., 0]<0] += 1  # 感觉没有必要,注释后仍能正常运行##################################################
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360    # 色度H是在0-360之间
        x[:, :, 1:][x[:, :, 1:]>1] = 1  # S和V的取值在0和1之间
        x[x<0] = 0  # H和S以及V的取值都是不小于0的
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255  # 返回RGB格式的图片

        # 调整目标框坐标
        box_data = np.zeros((len(box), 5)) # box保存了每张图片中所有真值框的信息
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx  # Xmin和Xmax
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy  # Ymin和Ymax
            if flip:  # 图像左右翻转 
                box[:, [0, 2]] = w - box[:, [2, 0]]  # Xmin,Xmax = w-Xmax, w-Xmin 注意这里的小细节
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]  # 每个真值框的宽度
            box_h = box[:, 3] - box[:, 1]  # 每个真值框的高度
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:, :4] > 0).any():
            return image_data, box_data 
            # 返回图片数据 多维数组(H,W,C)即(300, 300, 3)
            # 相应标签 (numLabel,xmin,ymin,xmax,ymax,category) 形状是(1, 5)
        else:
            return image_data, []

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines) # 对序列类型进行随机打乱
        lines = self.train_lines
        n = self.train_batches
        index = index % n
        while True:
            img, y = self.get_random_data(lines[index], self.image_size[0:2])
            if len(y)==0:
                continue
            boxes = np.array(y[:,:4],dtype=np.float32)  # 数据类型转换,注意剔出代表类别的数据列
            boxes[:,0] = boxes[:,0]/self.image_size[1]  # 标注信息的归一化,将像素坐标转为占图片整体的比例
            boxes[:,1] = boxes[:,1]/self.image_size[0]
            boxes[:,2] = boxes[:,2]/self.image_size[1]
            boxes[:,3] = boxes[:,3]/self.image_size[0]
            boxes = np.maximum(np.minimum(boxes,1),0)  # 将标注信息数据范围限制在0.0到1.0之间
            if ((boxes[:,3]-boxes[:,1])<=0).any() and ((boxes[:,2]-boxes[:,0])<=0).any():
                continue
            y = np.concatenate([boxes,y[:,-1:]],axis=-1)
            index = (index + 1) % n
            break
            
        img = np.array(img, dtype=np.float32)  # 转为float32类型,感觉没有必要，原数组的类型已经满足条件 
        tmp_inp = np.transpose(img-MEANS,(2,0,1))  # 将代表RGB通道的维度提前
        tmp_targets = np.array(y, dtype=np.float32)
        return tmp_inp, tmp_targets  # 返回图片(C,H,W) 以及相应标签信息(numLabel,xmin,ymin,xmax,ymax,category)


# DataLoader中collate_fn使用
# 这里就是用户自定义了批量化读取数据
# 其实这里可以使用PyTorch默认的collate函数
# 这里返回的是NumPy类型的数组
# PyTorch默认的是返回torch.Tensor类型,只需要在train.py中读取数据后稍微改动即可,如下所示
# images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
# images = Variable(images.type(torch.FloatTensor)).cuda()
# targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
# targets = [Variable((ann).type(torch.FloatTensor)).cuda() for ann in targets]


def ssd_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    bboxes = np.array(bboxes)
    return images, bboxes

