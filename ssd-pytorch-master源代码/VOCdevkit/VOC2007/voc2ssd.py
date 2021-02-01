'''
本程序用来生成三个文本文件,
分别是test.txt、train.txt、trainval.txt和val.txt
分别用来保存训练数据
(图片以及相应的xml格式的标签信息)的文件名,但不包括文件的扩展名.
这些图片按照一定的比例,
分别用来随机地构成测试集、
训练集、训练集和验证集、验证集.
'''

import os
import random 
random.seed(20200910)  # 设置随机数种子,用于重复随机的结果

xmlfilepath = r'./VOCdevkit/VOC2007/Annotations'        # 标签信息(xml文件)保存的位置
saveBasePath = r"./VOCdevkit/VOC2007/ImageSets/Main/"   # 分割之后保存信息的存储位置
 
trainval_percent = 0.9  # 表示用于训练和验证的数据数量占总体的比例
train_percent = 1.0     # 表示训练数据在训练集和验证集中所占的比例 

temp_xml = os.listdir(xmlfilepath)  # 指定目录下所有文件名字符串构成的列表
total_xml = []             # 用于保存所有xml文件的文件名字符串的列表
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num = len(total_xml)             # 所有数据的总量
tv = int(num*trainval_percent)   # 训练集和验证集的总量
tr = int(tv*train_percent)       # 训练数据的数量
trainval = random.sample(range(num), tv)  # 随机采样得到的训练集和验证集的下标索引
train = random.sample(trainval, tr)       # 随机获得训练集的下标索引
 
print("train and val size:(用于训练和验证的样本数量)",tv)
print("train size:(仅仅用于训练而不用于验证的样本数量)",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
 
for i  in range(num):  
    name = total_xml[i][:-4]+'\n'  # 出去末尾的4个字符,即文件扩展名.xml
    if i in trainval:              # 表示训练集和验证集
        ftrainval.write(name)  
        if i in train:             # 表示训练集
            ftrain.write(name)  
        else:  
            fval.write(name)       # 表示验证集
    else:  
        ftest.write(name)          # 表示测试集
  
ftrainval.close()   # 依次关闭这4个文件
ftrain.close()  
fval.close()  
ftest .close()
