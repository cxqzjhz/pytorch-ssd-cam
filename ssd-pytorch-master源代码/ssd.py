import cv2
import numpy as np
import colorsys
import os
import torch
from nets import ssd
import torch.backends.cudnn as cudnn
from utils.config import Config
from utils.box_utils import letterbox_image,ssd_correct_boxes
from PIL import Image,ImageFont, ImageDraw
from torch.autograd import Variable

MEANS = (104, 117, 123)
#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#--------------------------------------------#
class SSD(object):
    # 类方法处理的变量一定要是类变量，这里的_defaults就是类变量
    _defaults = {
        "model_path": "F:/Iris_SSD_small/ssd-pytorch-master/logs/Epoch50-Loc0.0260-Conf0.1510.pth",#'model_data/ssd_weights.pth',#模型训练好之后存放的位置
        "classes_path": 'model_data/voc_classes.txt', # 存放模型需要分的类所存放txt文件的位置
        "model_image_size" : (300, 300, 3), # 输入图片的大小
        "confidence": 0.5, # 可以忍受的目标检测的置信度
        "cuda": True,  # 是否使用显卡
    }

    # 类方法，参考链接: https://blog.csdn.net/leviopku/article/details/100745811?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159643686719724845007577%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=159643686719724845007577&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v3~pc_rank_v4-1-100745811.first_rank_ecpm_v3_pc_rank_v4&utm_term=%40classmethod&spm=1018.2118.3001.4187
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults: # 判断n是否在字典的关键字key中
            return cls._defaults[n] # 返回字典中与key=n相应的值value
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化SSD
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()  # ['FakeEye', 'LiveEye']
        self.generate()
        # print("林麻子,林祖泉.SSD初始化...")
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        # 计算总的种类
        self.num_classes = len(self.class_names) + 1

        # 载入模型
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ssd.get_ssd("test",self.num_classes)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))  # 通过hsv格式来调整不同类别对应边框的色度

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2]) # 获得图片的尺寸array([1330, 1330])
        # 在检测中，需要把原始图像转换为，与目标尺寸匹配的图像，保持等比例，其余部分用灰色填充。
        crop_img = np.array(letterbox_image(image, (self.model_image_size[0],self.model_image_size[1])))
        # 以下这行代码多余
        photo = np.array(crop_img,dtype = np.float64) # 类型转为dtype = np.float64
        # 图片预处理，归一化
        with torch.no_grad(): # 表示不进行求导 即使张量的requires_grad = True，也不求梯度 参考链接: https://blog.csdn.net/weixin_43178406/article/details/89517008?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159644470619195264563535%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=159644470619195264563535&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v3~pc_rank_v4-4-89517008.first_rank_ecpm_v3_pc_rank_v4&utm_term=with+torch.no_grad%28%29&spm=1018.2118.3001.4187
            photo = Variable(torch.from_numpy(np.expand_dims(np.transpose(crop_img-MEANS,(2,0,1)),0)).type(torch.FloatTensor))  # 将颜色通道对应的维度调整到前面
            # 标准化 通道维度调整到最前的维度 扩充维度 移动到GPU 
            # 【Python】 numpy.expand_dims的用法 参考链接: https://blog.csdn.net/qingzhuyuxian/article/details/90510203?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159645220719195188351469%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=159645220719195188351469&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-90510203.first_rank_ecpm_v3_pc_rank_v4&utm_term=np.expand_dims&spm=1018.2118.3001.4187

            if self.cuda:
                photo = photo.cuda()  # torch.Size([1, 3, 300, 300])
            preds = self.net(photo)  # torch.Size([1, 3, 200, 5])  1置信度+4位置信息
        # 
        top_conf = []  # 保存单张图片中所有预测框的置信度
        top_label = []  # 保存单张图片中所有预测框的物体类别
        top_bboxes = []  # 保存单张图片中所有预测框的位置信息
        for i in range(1, preds.size(1)): # 循环遍历21个类 i=0,1,2...,20 # 跳过背景所在的类,0代表背景
        # for i in range(1,preds.size(1)): for i in range(preds.size(1)):
            j = 0
            while preds[0, i, j, 0] >= self.confidence : # 对200个不同置信度的框的筛选
            # while preds[0, i, j, 0] >= self.confidence and j < preds.size(2): # 我的修改 ************************************************************************************************************************************************************************************************************************************************************************************
                score = preds[0, i, j, 0]
                # print("cxqcxq 陈旭旗陈旭旗self.class_names ：",self.class_names) # cxqcxq 陈旭旗陈旭旗self.class_names ： ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']  
                label_name = self.class_names[i-1]  # 有疑问 为什么不是 label_name = self.class_names[i]***************************************************************************************************************
                pt = (preds[0, i, j, 1:]).detach().numpy()
                coords = [pt[0], pt[1], pt[2], pt[3]]
                top_conf.append(score)
                top_label.append(label_name)
                top_bboxes.append(coords)
                j = j + 1
        # 将预测结果进行解码
        if len(top_conf)<=0: # 表示没有检测到任何东西，即全是背景
            return image
        top_conf = np.array(top_conf)
        top_label = np.array(top_label)
        top_bboxes = np.array(top_bboxes)
        top_xmin, top_ymin, top_xmax, top_ymax = \
            np.expand_dims(top_bboxes[:,0],-1),\
                np.expand_dims(top_bboxes[:,1],-1),\
                    np.expand_dims(top_bboxes[:,2],-1),\
                        np.expand_dims(top_bboxes[:,3],-1)

        # 去掉灰条
        boxes = ssd_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,\
            np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        font = ImageFont.truetype(
            font='model_data/simhei.ttf',
            size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]  
        # 8

        for i, c in enumerate(top_label):
            predicted_class = c
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)  
            label_size = draw.textsize(label, font)  # (240, 40)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])  # array([460, 654])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):  # 用于对边框加粗,起始可以使用width参数
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

