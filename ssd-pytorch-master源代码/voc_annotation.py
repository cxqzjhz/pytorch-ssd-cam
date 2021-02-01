"""
本程序的功能是根据之前得到的训练集、测试集、验证集
向三个文本文件2007_test.txt 2007_train.txt 2007_val.txt写入相关信息
每一行信息代表一个图片信息,首先是图片的绝对路径,
然后是图片中包含所有物体的信息,每个物体都有5个值,
分别是边框的xmin ymin xmax ymax以及代表类别的编号,
这5个值之间使用逗号分隔,
不同物体的信息之间以及图片绝对路径字符串之间使用空格分隔
"""

import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["FakeEye","LiveEye"]


def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC{0}/Annotations/{1}.xml'.format(year, image_id))
    tree = ET.parse(in_file) # 获得xml解析树
    root = tree.getroot()    # 获得xml解析树的根节点

    for obj in root.iter('object'): # 遍历每个object对象
        difficult = obj.find('difficult').text # 获得difficult的值
        category = obj.find('name').text       # 获得类别的名字
        if category not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(category)  # 获得该类别在列表中的下标索引
        xmlbox = obj.find('bndbox') # 获得该object的边框

        # 提取出边框的边界信息:xmin ymin xmax ymax,保存到元组b中
        b = (int(xmlbox.find('xmin').text), \
                int(xmlbox.find('ymin').text), \
                int(xmlbox.find('xmax').text), \
                int(xmlbox.find('ymax').text))

        info = " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
        # 向文件中写入信息,分别是四个坐标和代表类别的下表索引,这5个值用逗号分隔
        list_file.write(info)

wd = getcwd()

for year, image_set in sets: 
    # year, image_set = ('2007', 'train') ('2007', 'val') ('2007', 'test')
    
    # 获得相应数据集(训练集、测试集、验证集)的文件名(不含扩展名)列表
    image_ids = open('VOCdevkit/VOC{0}/ImageSets/Main/{1}.txt'.\
        format(year, image_set)).read().strip().split() 

    # 打开要写入的文本文件    
    with open('{0}_{1}.txt'.format(year, image_set), 'w') as list_file:

        # 遍历相应数据集(训练集、测试集、验证集)的文件名(不含扩展名)
        for image_id in image_ids:
            
            # 在文本文件中写入jpg图片文件的绝对路径
            list_file.write('{0}/VOCdevkit/VOC{1}/JPEGImages/{2}.jpg'.\
                format(wd, year, image_id))
            
            # 向指定文件中写入标签信息,即对于每个物体,
            # 写入边框的四个坐标以及代表类别的编号,
            # 这五个数之间使用逗号分隔,
            # 这五个数与之前的图片文件路径使用空格分隔
            convert_annotation(year, image_id, list_file)
            # 每一个图片的相关信息占用一行
            list_file.write('\n')