import torch
import numpy as np
from PIL import Image
def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin  # 中心点的横纵坐标减去一半的宽度高度
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax  # 中心点的横纵坐标加上一半的宽度高度


def center_size(boxes):
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    A = box_a.size(0)  # 真值框的数量 1       torch.Size([1, 4])
    B = box_b.size(0)  # 先验框的数量 8732    torch.Size([8732, 4])
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))  # torch.Size([1, 8732, 2])  # xmax, ymax 右下边界
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))  # torch.Size([1, 8732, 2])  # xmin, ymin 左上边界
    inter = torch.clamp((max_xy - min_xy), min=0)  # torch.Size([1, 8732, 2])
    # 计算先验框和所有真实框的重合面积
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    # box_a  torch.Size([1, 4])
    # box_b  torch.Size([8732, 4])
    inter = intersect(box_a, box_b)  # torch.Size([1, 8732])
    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2]-box_a[:, 0]) * 
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]  torch.Size([1, 8732])
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]  torch.Size([1, 8732])
    # 求IOU
    union = area_a + area_b - inter  # torch.Size([1, 8732])
    return inter / union  # [A,B]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    '''对一个batch中的特定的一张图片处理,将其先验框和真实框进行匹配,保证每个真实框都得到匹配.
    一般真实框都匹配和它重合最高的先验框,除非有真实框未得到匹配,那么将这个先验框匹配到该真实框.
    '''
    # threshold, # 0.5
    # truths, # 单个图片中所有物体的位置信息
    # priors, # torch.Size([8732, 4])
    # variances, # [0.1, 0.2]
    # labels, # 单个图片的所有物体的类别信息
    # loc_t, # torch.Size([4, 8732, 4])
    # conf_t, # torch.Size([4, 8732])
    # idx, # 这是依次遍历batchsize的长度
   
    # 计算所有的先验框和真实框的重合程度
    overlaps = jaccard(
        truths,
        point_form(priors)
    )  # torch.Size([1, 8732])
    # 所有真实框和先验框的最好重合程度
    # [truth_box,1]
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # best_prior_overlap  torch.Size([1, 1])
    # best_prior_idx  torch.Size([1, 1])
    best_prior_idx.squeeze_(1)  # torch.Size([1])
    best_prior_overlap.squeeze_(1)  # torch.Size([1])
    # 所有先验框和真实框的最好重合程度
    # [1,prior]
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    # torch.Size([1, 8732])
    # torch.Size([1, 8732])
    best_truth_idx.squeeze_(0)  # torch.Size([8732])
    best_truth_overlap.squeeze_(0)  # torch.Size([8732])
    # 找到与真实框重合程度最好的先验框，用于保证每个真实框都要有对应的一个先验框
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # 保证每一个真值框都有一个先验框与之对应,IOU=2
    # 对best_truth_idx内容进行设置
    for j in range(best_prior_idx.size(0)):  # 遍历真值框数量
        best_truth_idx[best_prior_idx[j]] = j  
        # 这行代码的意思是保证每个真值框都有一个先验框与之对应,
        # 尽管该先验框与其他真值框的IOU可能会更大,依然将这个先验框对应到该真值框
    
    # 找到每个先验框重合程度最好的真实框
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]  torch.Size([8732, 4])
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]  torch.Size([8732])  
    # 加1的原因是0作为背景 0背景,1FakeEye,2LiveEye
    
    # 如果重合程度小于threhold则认为是背景
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)  # torch.Size([8732, 4])
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn  torch.Size([4, 8732, 4])
    conf_t[idx] = conf  # [num_priors] top class label for each prior  torch.Size([8732])


def encode(matched, priors, variances):
    # priors 中心点的横坐标位置 # 宽度和高度 (占整个图片尺寸的比例)
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]  # 中心点的相对先验框左上角点的偏移
    g_cxcy /= (variances[0] * priors[:, 2:])  # 中心点相对先验框尺寸的相对偏移
    # g_cxcy  torch.Size([8732, 2])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]  # 真实框的宽度高度相对于先验框的比例
    g_wh = torch.log(g_wh) / variances[1]  # torch.Size([8732, 2])
    return torch.cat([g_cxcy, g_wh], 1) 

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):  # torch.Size([34928, 3])
    x_max = x.data.max()  # 试试 x_max = x.max()
    # x_max = x.max()  # 换成这行代码也能正常运行
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    # boxes  torch.Size([11, 4])
    # scores  torch.Size([11])
    # overlap  0.45
    # top_k  200
    keep = scores.new(scores.size(0)).zero_().long()  # torch.Size([11]) 用于记录该保留下的框的索引
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]  # xmin
    y1 = boxes[:, 1]  # ymin
    x2 = boxes[:, 2]  # xmax
    y2 = boxes[:, 3]  # ymax
    area = torch.mul(x2 - x1, y2 - y1)  # 所有框的面积
    v, idx = scores.sort(0)  # 对置信度进行升序排序
    idx = idx[-top_k:]  # 获得前top_k个最高置信度的索引
    # xx1 = boxes.new() 
    xx1 = boxes.new()  # 其实这里可以不用新创建,只需要后续改成 xx1 = torch.index_select(x1, 0, idx)即可
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0 # 用于记录筛选出来的框的数量
    while idx.numel() > 0:
        i = idx[-1]  # 获得剩余所有框中置信度最高的框的索引
        keep[count] = i  # 选出当前剩余中的框置信度最高的框的索引
        count += 1  # 已选出的框的数量增加1
        if idx.size(0) == 1:  # 所有框都已经遍历完毕
            break
        idx = idx[:-1]  # 置信度最高的索引从待选择部分移除
        torch.index_select(x1, 0, idx, out=xx1)
        # xx1 = torch.index_select(x1, 0, idx) # 使用这行代码的话前面就不需要新创建
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[i])  # 获得矩形重合部分的xmin
        yy1 = torch.clamp(yy1, min=y1[i])  # 获得矩形重合部分的ymin
        xx2 = torch.clamp(xx2, max=x2[i])  # 获得矩形重合部分的xmax
        yy2 = torch.clamp(yy2, max=y2[i])  # 获得矩形重合部分的ymax
        w.resize_as_(xx2)  # 这两行代码多余
        h.resize_as_(yy2)  # 这两行代码多余
        w = xx2 - xx1  # 获得矩形重合部分的宽度
        h = yy2 - yy1  # 获得矩形重合部分的高度
        w = torch.clamp(w, min=0.0)  # 这两行代码用于处理不重叠的情形
        h = torch.clamp(h, min=0.0)  # 这两行代码用于处理不重叠的情形
        inter = w*h  # 计算和当前置信度最高框的重叠部分面积
        rem_areas = torch.index_select(area, 0, idx)  # 计算这些剩余部分框各自的面积
        union = (rem_areas - inter) + area[i]  # 计算这些剩余部分框各自和当前置信度最高框合并之后的面积
        IoU = inter/union  # 计算剩余所有框各自与当前最高置信度框的交并比
        idx = idx[IoU.le(overlap)]  # 移除所有重叠过高的框,保留低重叠的框
    return keep, count  # 非极大抑制操作之后该保留的框的索引下标,以及该保留的这些框的数量

def letterbox_image(image, size):
    iw, ih = image.size  # 1330, 1330
    w, h = size  # 300, 300
    scale = min(w/iw, h/ih)  # 0.22556390977443608
    nw = int(iw*scale)  # 300
    nh = int(ih*scale)  # 300

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def ssd_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    # top  (2, 1)  ndarray  
    # left  (2, 1)  ndarray  
    # bottom  (2, 1)  ndarray  
    # right  (2, 1)  ndarray  
    # input_shape  (2,)  ndarray  array([300, 300])
    # image_shape  (2,)  ndarray  array([1330, 1330])
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    print(np.shape(boxes))
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes