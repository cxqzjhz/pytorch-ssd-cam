
'''
该程序功能,
将检测数据写到文件夹input/detection-results/下,
每张图片对应一个txt文件,
txt文件内容是每个预测框占用一行,
每一行分别是类别名称,置信度,xmin,ymin,xmax,ymax,分别用空格分隔.
同时将测试所使用的图片以jpg格式保存到input\images-optional目录下
'''

from ssd import SSD
from PIL import Image
from utils.box_utils import letterbox_image,ssd_correct_boxes
from torch.autograd import Variable
import torch
import numpy as np
import os
MEANS = (104, 117, 123)
class mAP_SSD(SSD):  # 继承子类SSD
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        self.confidence = 0.05
        f = open("./input/detection-results/"+image_id+".txt","w") 
        image_shape = np.array(np.shape(image)[0:2])  # array([480, 640])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0],self.model_image_size[1])))  # 形状(300, 300, 3)
        photo = np.array(crop_img,dtype = np.float64)  # 形状(300, 300, 3)
        # 图片预处理，归一化
        with torch.no_grad():
            photo = Variable(torch.from_numpy(np.expand_dims(np.transpose(crop_img-MEANS,(2,0,1)),0)).type(torch.FloatTensor))
            if self.cuda:
                photo = photo.cuda()
            preds = self.net(photo)  # torch.Size([1, 3, 200, 5])
        top_conf = []
        top_label = []
        top_bboxes = []

        # for i in range(1, preds.size(1)):  # 跳过背景所代表的类
        for i in range(1,preds.size(1)):  # 遍历每一个类  
            # if i == 0:
            #     continue # 跳过背景所代表的类
            j = 0
            while preds[0, i, j, 0] >= self.confidence:
                score = preds[0, i, j, 0]
                label_name = self.class_names[i-1]  # 注意在preds中0代表背景,所以需要i-1
                pt = (preds[0, i, j, 1:]).detach().numpy()
                coords = [pt[0], pt[1], pt[2], pt[3]]
                top_conf.append(score)
                top_label.append(label_name)
                top_bboxes.append(coords)
                j = j + 1
        # 将预测结果进行解码
        if len(top_conf)<=0:
            return image
        top_conf = np.array(top_conf)
        top_label = np.array(top_label)
        top_bboxes = np.array(top_bboxes)
        top_xmin, top_ymin, top_xmax, top_ymax = \
            np.expand_dims(top_bboxes[:,0],-1),\
            np.expand_dims(top_bboxes[:,1],-1),\
            np.expand_dims(top_bboxes[:,2],-1),\
            np.expand_dims(top_bboxes[:,3],-1)  # top_bboxes[:,3:4] 

        # 去掉灰条
        boxes = ssd_correct_boxes(
            top_ymin,top_xmin,top_ymax,top_xmax,
            np.array([self.model_image_size[0],self.model_image_size[1]]),
            image_shape)


        for i, c in enumerate(top_label):
            predicted_class = c
            score = str(float(top_conf[i]))

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" %\
                (predicted_class, score[:6], str(int(left)), \
                    str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

ssd = mAP_SSD()  # 这里会自动执行子类的初始化,即执行子类的__init__()方法

# 获得图片文件名,不包含文件扩展名,以字符串的形式保存到列表中
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")  # 创建目录
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")  # 创建目录
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")  # 创建目录


for image_id in image_ids:
    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    ###########
    if os.path.exists(image_path):  # 这里之所以要判断一下是因为本人使用的图片有jpg格式和bmp格式
        # print(myPath)
        pass
    else:
        image_path = image_path[:-3] + "bmp"  # 该图片不是jpg格式,而是bmp格式
        # print(myPath)
    #############
    image = Image.open(image_path)
    image.save("./input/images-optional/"+image_id+".jpg")  # 将bmp图片或者jpg图片统一保存为jpg格式
    ssd.detect_image(image_id,image)
    print(image_id," done!")
    

print("Conversion completed!")
