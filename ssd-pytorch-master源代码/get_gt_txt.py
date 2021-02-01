#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#

'''
该程序功能,
将真值数据写到文件夹input/ground-truth下,
每张图片对应一个txt文件,
txt文件内容是每个真实框占用一行,
每一行分别是:   类别名称,xmin,ymin,xmax,ymax,分别用空格分隔.
'''

import sys
import os
import glob
import xml.etree.ElementTree as ET

# 获得图片文件名,不包含文件扩展名,以字符串的形式保存到列表中
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")  # 创建目录
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")  # 创建目录

for image_id in image_ids:  # 遍历图片文件名,不包含文件扩展名
    with open("./input/ground-truth/"+image_id+".txt", "w") as new_f:
        root = ET.parse("VOCdevkit/VOC2007/Annotations/"+image_id+".xml").getroot()
        for obj in root.findall('object'):
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
                if int(difficult)==1:
                    continue
            obj_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text
            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
            # 为图片中的每个物体(真值框)写一行数据,分别是: 类别名,xmin,ymin,xmax,ymax
print("Conversion completed!")
