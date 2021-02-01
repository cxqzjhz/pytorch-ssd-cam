from nets.ssd import get_ssd
from nets.ssd_training import Generator,MultiBoxLoss
from torch.utils.data import DataLoader
from utils.dataloader import ssd_dataset_collate, SSDDataset
from utils.config import Config
from torchsummary import summary
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init



# 设置随机数种子
import numpy as np
import random
import torch
import os
transfer_learning = True  # False  True  # 此变量用来设置是否使用迁移学习
seed = 20200910
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# You need to set the init function of the worker(s) to be fed to the DataLoader:
r""" 
def _init_fn(worker_id):
    np.random.seed(seed)
trainloader = DataLoader(trainset, batch_size=batch_size, 
    shuffle=True, num_workers=num_workers,   
    pin_memory=True, worker_init_fn=_init_fn)

"""




def adjust_learning_rate(optimizer, lr, gamma, step):
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    # ------------------------------------#
    #   先冻结一部分权重训练
    #   后解冻全部权重训练
    #   先大学习率
    #   后小学习率
    # ------------------------------------#
    lr = 5e-4
    freeze_lr = 1e-4
    Cuda = True

    Start_iter = 0
    Freeze_epoch = 25
    Epoch = 50

    Batch_size = 4
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True
    
    model = get_ssd("train",Config["num_classes"])  # get_ssd("train", 3)
    
    # 是否使用迁移学习
    if transfer_learning:
        #-------------------------------------------#
        #   权值文件的下载请看README
        #-------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load("model_data/ssd_weights.pth", map_location=device)
        pretrained_dict = {
                            k: v for k, v in pretrained_dict.items() \
                            if np.shape(model_dict[k]) ==  np.shape(v)\
                        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    

    print('Finished!')

    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)  # 本人的机器只有一个GPU故可以将这行代码注释掉
        # cudnn.benchmark = True # 注释掉,取消随机性
        net = net.cuda()

    annotation_path = '2007_train.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
    # np.random.seed(seed)
    np.random.shuffle(lines)
    # np.random.seed(None)
    num_train = len(lines)

    if Use_Data_Loader:
        train_dataset = SSDDataset(
            lines[:num_train], \
            ( Config["min_dim"], Config["min_dim"] )
        )
        gen = DataLoader(train_dataset, \
                        batch_size=Batch_size, \
                        num_workers=8, \
                        pin_memory=True,\
                        drop_last=True, \
                        collate_fn=ssd_dataset_collate)
                        # collate_fn=None)
    else:
        gen = Generator(Batch_size, lines,
                        (Config["min_dim"], Config["min_dim"]), Config["num_classes"]).generate()

    criterion = MultiBoxLoss(Config['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, Cuda)
    # criterion = MultiBoxLoss(2+1, 0.5, True, 0, True, 3, 0.5,False, True)

    epoch_size = num_train // Batch_size


    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    
    if True:
    # if False:
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if transfer_learning:
            for param in model.vgg.parameters():
                param.requires_grad = False  # 使用迁移学习

        optimizer = optim.Adam(net.parameters(), lr=lr)
        for epoch in range(Start_iter,Freeze_epoch):
            if epoch%10==0:
                adjust_learning_rate(optimizer,lr,0.9,epoch)
            loc_loss = 0
            conf_loss = 0
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_size:
                    break
                images, targets = batch[0], batch[1]  # (4, 3, 300, 300)  形状(4, 1, 5)
                # images 返回图片数据 多维数组(H,W,C)即(300, 300, 3) 
                # 由于自动批量化,这里形状是(4, 3, 300, 300)
                # targets 相应标签 (numLabel,xmin,ymin,xmax,ymax,category) 
                # 形状是(1, 5), 其中1代表图片只有一个物体
                # 由于自动批量化,这里形状是(4, 1, 5)
                # 注意这里的targets是个numpy多维数组,它的元素不一定同质
                # 因为同一个batch中不同的图片所含的物体数量不一定相同,
                # 当物体数量不相同时,dtype是object类型,但在当前的实验下 
                # 每张图片中只包含一个眼睛,
                # 因此是同质的,所以形状是(4, 1, 5),dtype=float32
                with torch.no_grad():
                    if Cuda:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                        # images = Variable(images.type(torch.FloatTensor)).cuda()
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                        # targets = [Variable((ann).type(torch.FloatTensor)).cuda() for ann in targets]
                    else:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                # 前向传播
                out = net(images)  
                # 长度为3的元组,三个元素的形状分别是 torch.Size([4, 8732, 4]) torch.Size([4, 8732, 3]) torch.Size([8732, 4])
                # 清零梯度
                optimizer.zero_grad()
                # 计算loss
                # targets是一个长度为4的列表形状分别是 torch.Size([1, 5])  torch.Size([1, 5]) torch.Size([1, 5]) torch.Size([1, 5])
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                # 反向传播
                loss.backward()
                optimizer.step()
                # 加上
                loc_loss += loss_l.item()
                conf_loss += loss_c.item()

                # print('\nEpoch:'+ str(epoch+1) + '/' + str(Freeze_epoch))
                print('Epoch: {0:->10,}/{1:-<10,}'.format(epoch+1,Freeze_epoch))
                # print('iter:' + str(iteration) + '/' + str(epoch_size) + ' || Loc_Loss: %.4f || Conf_Loss: %.4f ||' % (loc_loss/(iteration+1),conf_loss/(iteration+1)), end=' ')
                # print()
                print('iter:{0:>6}/{1:<6} || Loc_Loss: {2:^10.4f} || Conf_Loss: {3:^10.4f} ||'.format(\
                    iteration,epoch_size,\
                    loc_loss/(iteration+1),\
                    conf_loss/(iteration+1))
                    )
                # break

            print('Saving state, iter:', str(epoch+1))
            # torch.save(model.state_dict(), 'logs/Epoch{0:}-Loc{1:.4f}-Conf{2:.4f}.pth'.format(epoch+1,loc_loss/(iteration+1),conf_loss/(iteration+1))) 
            torch.save(model.state_dict(), \
                'logs/Epoch{0:}-Loc{1:.4f}-Conf{2:.4f}.pth'\
                .format(epoch+1,\
                    loc_loss/(iteration+1),\
                    conf_loss/(iteration+1)
                    )
            ) 

    if True:
        # ------------------------------------#
        #   全部解冻训练
        # ------------------------------------#
        for param in model.vgg.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(net.parameters(), lr=freeze_lr)
        for epoch in range(Freeze_epoch,Epoch):
            if epoch%10==0:
                adjust_learning_rate(optimizer,freeze_lr,0.9,epoch)
            loc_loss = 0
            conf_loss = 0
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_size:
                    break
                images, targets = batch[0], batch[1]
                with torch.no_grad():
                    if Cuda:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                    else:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                # 前向传播
                out = net(images)
                # 清零梯度
                optimizer.zero_grad()
                # 计算loss
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                # 反向传播
                loss.backward()
                optimizer.step()
                # 加上
                loc_loss += loss_l.item()
                conf_loss += loss_c.item()

                # print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
                print('Epoch: {0:->10,}/{1:-<10,}'.format(epoch+1,Epoch))
                # print('iter:' + str(iteration) + '/' + str(epoch_size) + ' || Loc_Loss: %.4f || Conf_Loss: %.4f ||' % (loc_loss/(iteration+1),conf_loss/(iteration+1)), end=' ')
                print('iter:{0:>6}/{1:<6} || Loc_Loss: {2:^10.4f} || Conf_Loss: {3:^10.4f} ||'.format(\
                    iteration,epoch_size,\
                    loc_loss/(iteration+1),\
                    conf_loss/(iteration+1))
                    )
                # break

            print('Saving state, iter:', str(epoch+1))
            # torch.save(model.state_dict(), 'logs/Epoch%d-Loc%.4f-Conf%.4f.pth'%((epoch+1),loc_loss/(iteration+1),conf_loss/(iteration+1)))
            torch.save(model.state_dict(), \
                'logs/Epoch{0:}-Loc{1:.4f}-Conf{2:.4f}.pth'\
                .format(epoch+1,\
                    loc_loss/(iteration+1),\
                    conf_loss/(iteration+1)
                    )
            ) 
