import torch
import argparse
import cv2
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from torchvision import models, transforms
from ssd import SSD
from nets.ssd import get_ssd
from PIL import Image, ImageDraw
from utils.box_utils import letterbox_image,ssd_correct_boxes
MEANS = (104, 117, 123)



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
        xx1 = torch.index_select(x1, 0, idx)
        # xx1 = torch.index_select(x1, 0, idx) # 使用这行代码的话前面就不需要新创建
        yy1 = torch.index_select(y1, 0, idx)
        xx2 = torch.index_select(x2, 0, idx)
        yy2 = torch.index_select(y2, 0, idx)
        xx1 = torch.clamp(xx1, min=x1[i].item())  # 获得矩形重合部分的xmin
        yy1 = torch.clamp(yy1, min=y1[i].item())  # 获得矩形重合部分的ymin
        xx2 = torch.clamp(xx2, max=x2[i].item())  # 获得矩形重合部分的xmax
        yy2 = torch.clamp(yy2, max=y2[i].item())  # 获得矩形重合部分的ymax
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


def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def detect_forward(loc_data, conf_data, prior_data):  # class Detect(Function):
    # # loc_data preds torch.Size([1, 8732, 4])
    # # conf_data  # torch.Size([1, 8732, 3]) 
    # # prior_data torch.Size([8732, 4])
    loc_data = loc_data.cpu()
    conf_data = conf_data.cpu()
    num = loc_data.size(0)  # batch size 1
    num_priors = prior_data.size(0)  # 8732
    output = torch.zeros(num, 3, 200, 5)  # torch.Size([1, 3, 200, 5])
    conf_preds = conf_data.view(num, num_priors, 3).transpose(2, 1)  # torch.Size([1, 3, 8732])
    # 对每一张图片进行处理
    for i in range(num):
        # 对先验框解码获得预测框
        decoded_boxes = decode(loc_data[i], prior_data, [0.1, 0.2])  # torch.Size([8732, 4])
        conf_scores = conf_preds[i].clone()  # torch.Size([3, 8732])

        for cl in range(1, 3):  # 遍历1到2,因为0代表背景
            # 对每一类进行非极大抑制
            c_mask = conf_scores[cl].gt(0.01)  # 获取正样本的索引  torch.Size([8732])
            scores = conf_scores[cl][c_mask]  # 获取所有正样本的置信度分数  torch.Size([11])
            if scores.size(0) == 0:
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)  # torch.Size([8732, 4])
            boxes = decoded_boxes[l_mask].view(-1, 4)  # torch.Size([11, 4]) 获取所有正样本的边框
            # 进行非极大抑制
            ids, count = nms(boxes, scores, 0.45, 200)
            output[i, cl, :count] = \
                torch.cat((scores[ids[:count]].unsqueeze(1),
                        boxes[ids[:count]]), 1)
    return output  # torch.Size([1, 3, 200, 5])  1置信度+4位置信息
    
    



class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        # FeatureExtractor(model.layer4, ["2"])
        self.model = model  # model.layer4
        self.target_layers = target_layers  # ["2"]
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)  # torch.Size([1, 2048, 7, 7])

    def __call__(self, x):  # torch.Size([1, 1024, 14, 14])
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            # '0'、 '1'、 '2'
            x = module(x)
            if name in self.target_layers:  # ["2"]
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x   # 单个元素的列表torch.Size([1, 2048, 7, 7]) torch.Size([1, 2048, 7, 7])

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        # ModelOutputs(model, model.layer4, ["2"])
        self.model = model  # model
        self.feature_module = feature_module  # model.layer4
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)
        # FeatureExtractor(model.layer4, ["2"])

    def get_gradients(self):
        return self.feature_extractor.gradients  # 只有一个元素列表类型 torch.Size([1, 2048, 7, 7])

    def __call__(self, x):
        # target_activations = []  # 这行代码没有意义
        for name, module in self.model._modules.items():  # 遍历有序字典
        # 'conv1' 'bn1' 'relu' 'maxpool' 'layer1' 
        # 'layer2' 'layer3' 'layer4'  'avgpool' 'fc'
            if module == self.feature_module:  # model.layer4
                target_activations, x = self.feature_extractor(x) 
                # torch.Size([1, 1024, 14, 14]) -> torch.Size([1, 2048, 7, 7])
            elif "avgpool" in name.lower():  # 'avgpool'
                x = module(x)  # torch.Size([1, 2048, 7, 7]) -> torch.Size([1, 2048, 1, 1])
                x = x.view(x.size(0),-1)  # torch.Size([1, 2048])
            else:
                x = module(x)

        return target_activations, x  # 列表torch.Size([1, 2048, 7, 7]), torch.Size([1, 1000])

def preprocess_image(img):
    '''将numpy的(H, W, RGB)格式多维数组转为张量后再进行指定标准化,最后再增加一个batchsize维度后返回'''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def show_cam_on_image(img, mask):
    '''将mask图片转化为热力图,叠加到img上,再返回np.uint8格式的图片.'''
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class GradCam:
    def __init__(self, model, use_cuda):
        # GradCam(model=model, feature_module=model.layer4, \
        #                target_layer_names=["2"], use_cuda=args.use_cuda)
        self.model = model  # model
        # self.feature_module = feature_module  # model.layer4
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)
        # ModelOutputs(model, model.layer4, ["2"])
    def forward(self, input_img):  # 似乎这个方法没有使用到,注释掉之后没有影响,没有被执行到
        print("林麻子".center(50,'-'))  # 这行打印语句用来证明,该方法并没有被调用执行.
        return self.model(input_img)  

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()  # torch.Size([1, 3, 224, 224])
        
        loc, conf, priors  = self.model(input_img)
        conf = (nn.Softmax(dim=-1))(conf)
        output = detect_forward(loc, conf, priors)  # torch.Size([1, 3, 200, 5])  1置信度+4位置信息
        scores = output[:,1:,:,0]
        features = self.model.feature_maps4cxq[1]
        # features = torch.Size([1, 1024, 19, 19])
        features.retain_grad()
        # output = [torch.Size([1, 38, 38, 12]),torch.Size([1, 19, 19, 18]),torch.Size([1, 10, 10, 18]),
        # torch.Size([1, 5, 5, 18]),torch.Size([1, 3, 3, 12]),torch.Size([1, 1, 1, 12])]
        
        # features, output = self.extractor(input_img)  # 保存中间特征图的列表, 以及网络最后输出的分类结果
        # 列表[torch.Size([1, 2048, 7, 7])], 张量:torch.Size([1, 1000])
        # if target_category == None:
        #     target_category = np.argmax(output.cpu().data.numpy())  # 多维数组展平后最大值的索引
        #     # <class 'numpy.int64'>  243

        # one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)  # 独热编码,shape:(1, 1000)
        # one_hot[0,target_category] = 1  # 独热编码  shape (1, 1000) # one_hot[0][target_category] = 1
        # one_hot = torch.from_numpy(one_hot).requires_grad_(False)  # torch.Size([1, 1000]) # requires_grad_(True)
        # if self.cuda:
        #     one_hot = one_hot.cuda()
        mask = scores > 0.5
        loss = torch.sum(scores[mask])  # tensor(9.3856, grad_fn=<SumBackward0>) one_hot = torch.sum(one_hot * output)

        # self.feature_module.zero_grad()  # 将模型的所有参数的梯度清零.
        self.model.zero_grad()  # 将模型的所有参数的梯度清零.
        loss.backward()  # one_hot.backward(retain_graph=True)  

        grads_val = features.grad.cpu().data.numpy()  # shape:(1, 2048, 7, 7)  # 顾名思义,梯度值
        # 注: self.extractor.get_gradients()[-1]返回保存着梯度的列表,[-1]表示最后一项,即最靠近输入的一组特征层上的梯度
        target = features  # torch.Size([1, 2048, 7, 7])  列表中的最后一项,也是唯一的一项,特征图
        target = target.cpu().data.numpy()[0, :]  # shape: (2048, 7, 7)

        weights = np.mean(grads_val, axis=(2, 3))[0, :]  # shape: (2048,)  计算每个特征图上梯度的均值,以此作为权重
        cam = np.zeros(target.shape[1:], dtype=np.float32)  # 获得零矩阵 shape: (7, 7)

        for i, w in enumerate(weights):  # 迭代遍历该权重
            cam += w * target[i, :, :]   # 使用该权重,对特征图进行线性组合

        cam = np.maximum(cam, 0)  # shape: (7, 7) # 相当于ReLU函数
        # print(type(input_img.shape[3:1:-1]),'cxq林麻子cxq',input_img.shape[3:1:-1])
        # print(type(input_img.shape[2:]),'cxq林麻子cxq',input_img.shape[2:])
        cam = cv2.resize(cam, input_img.shape[3:1:-1])  # shape: (224, 224) # 这里要留意传入的形状是(w,h) 所以这里切片的顺序是反过来的
        cam = cam - np.min(cam)  # shape: (224, 224)  # 以下两部是做归一化
        cam = cam / np.max(cam)  # shape: (224, 224)  # 归一化,取值返回是[0,1]
        return cam  # shape: (224, 224) 取值返回是[0,1]


class GuidedBackpropReLU(Function):
    '''特殊的ReLU,区别在于反向传播时候只考虑大于零的输入和大于零的梯度'''
    
    '''
    @staticmethod
    def forward(ctx, input_img):  # torch.Size([1, 64, 112, 112])
        positive_mask = (input_img > 0).type_as(input_img)  # torch.Size([1, 64, 112, 112])
        # output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        output = input_img * positive_mask  # 这行代码和上一行的功能相同
        ctx.save_for_backward(input_img, output)
        return output  # torch.Size([1, 64, 112, 112])
    '''
    # 上部分定义的函数功能和以下定义的函数一致
    @staticmethod
    def forward(ctx, input_img):  # torch.Size([1, 64, 112, 112])
        output = torch.clamp(input_img, min=0.0)
        # print('函数中的输入张量requires_grad',input_img.requires_grad)
        ctx.save_for_backward(input_img, output)
        return output  # torch.Size([1, 64, 112, 112])

    @staticmethod
    def backward(ctx, grad_output):  # torch.Size([1, 2048, 7, 7])
        input_img, output = ctx.saved_tensors  # torch.Size([1, 2048, 7, 7]) torch.Size([1, 2048, 7, 7])
        # grad_input = None  # 这行代码没作用
        positive_mask_1 = (input_img > 0).type_as(grad_output)  # torch.Size([1, 2048, 7, 7])  输入的特征大于零
        positive_mask_2 = (grad_output > 0).type_as(grad_output)  # torch.Size([1, 2048, 7, 7])  梯度大于零
        # grad_input = torch.addcmul(
        #                             torch.zeros(input_img.size()).type_as(input_img),
        #                             torch.addcmul(
        #                                             torch.zeros(input_img.size()).type_as(input_img), 
        #                                             grad_output,
        #                                             positive_mask_1
        #                             ), 
        #                             positive_mask_2
        # )
        grad_input = grad_output * positive_mask_1 * positive_mask_2  # 这行代码的作用和上一行代码相同
        return grad_input


class GuidedBackpropReLU_Module_by_cxq(nn.Module):
    def __init__(self):
        super(GuidedBackpropReLU_Module_by_cxq, self).__init__()

    def forward(self, input):
        return GuidedBackpropReLU.apply(input)  

    def extra_repr(self):
        '''该方法用于打印信息'''
        return '我是由cxq实现的用于自定义GuidedBackpropReLU的网络模块...'




class GuidedBackpropReLUModel:
    '''相对于某个类别(默认是最大置信度对应的类别)的置信度得分,计算输入图片上的梯度,并返回'''
    def __init__(self, model, use_cuda):  
        # GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            '''递归地将模块内的relu模块替换掉用户自己定义的GuidedBackpropReLU模块 '''
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':  # module对象所属的类,该类的名称
                    # print('成功替换...')  # 验证确实得到了替换
                    # module_top._modules[idx] = GuidedBackpropReLU.apply  # 这是原始代码所使用的方式
                    module_top._modules[idx] = GuidedBackpropReLU_Module_by_cxq()  # 这是本人cxq改进的方式
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    # def forward(self, input_img):
    #     return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        '''相对于某个类别(默认是最大置信度对应的类别)的置信度得分,计算输入图片上的梯度,并返回'''
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)  # torch.Size([1, 3, 224, 224])
        output = self.model(input_img)[1]  # torch.Size([1, 8732, 3])
 
        one_or_two = False  # True, False
        if one_or_two:
            # 方式1:  分数最大的框
            loss = torch.max(output)
        else: 
            # 方式2:  分数最大的k(其中:k=200)个框
            scores = output.view(-1)
            values, indices = torch.topk(scores, k=200, dim=0, largest=True, sorted=True, out=None)
            loss = torch.sum(scores[indices])


        loss.backward()  # one_hot.backward(retain_graph=True)

        img_grad = input_img.grad.cpu().data.numpy()  # shape (1, 3, 224, 224)
        img_grad = img_grad[0, :, :, :]  # shape (3, 224, 224)

        return img_grad  # shape (3, 224, 224)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='3.jpg',  # './examples/1.jpg','2.jpg' './examples/both.png'  3.jpg
                        help='Input image path')  # default='./examples/both.png',
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    '''先作标准化处理,然后做变换y=0.1*x+0.5,限定[0,1]区间后映射到[0,255]区间'''
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()  
    # 默认情况下: args.image_path = './examples/both.png', 
    # 默认情况下: args.use_cuda = False, 

    image = Image.open(args.image_path)
    # image.show()
    # image_size = image.size
    iw, ih = image.size  # 640, 480
    w, h = 300, 300
    scale = min(w/iw, h/ih)  # 0.46875
    nw = int(iw*scale)  # 300
    nh = int(ih*scale)  # 225
    box = [(w-nw)//2, (h-nh)//2, nw+(w-nw)//2, nh+(h-nh)//2]  



    ssd = SSD()
    model = get_ssd("train", 3)  # ssd.net
    model.load_state_dict(torch.load(
        "F:/Iris_SSD_small/ssd-pytorch-master/logs/Epoch50-Loc0.0260-Conf0.1510.pth", 
        map_location=torch.device('cuda' )
        )
    )
    ssd.net = model.eval()
    ssd.net = torch.nn.DataParallel(ssd.net)
    ssd.net = ssd.net.cpu()  # ****
    model = ssd.net.module

    ########################################################################################################
    
    
    # model = models.resnet50(pretrained=True)
    model.phase = 'Grad-CAM'
    grad_cam = GradCam(model=model, use_cuda=args.use_cuda)


    image = Image.open(args.image_path)
    image = image.convert('RGB')
    image_shape = np.array(np.shape(image)[0:2]) # 获得图片的尺寸
    crop_img = np.array(letterbox_image(image, (300,300)))  # (300, 300, 3)
    # photo = np.array(crop_img,dtype = np.float64) # 类型转为dtype = np.float64

    photo = torch.from_numpy(np.expand_dims(np.transpose(crop_img-MEANS,(2,0,1)),0))\
        .type(torch.FloatTensor).requires_grad_(True)  # 将颜色通道对应的维度调整到前面 torch.Size([1, 3, 300, 300])
    photo = photo.requires_grad_().cpu()  # .cpu()  # 范围是0-255  torch.Size([1, 3, 300, 300])





    img = cv2.imread(args.image_path, 1)  # 读取图片文件 (H, W, BGR)
    # If set, always convert image to the 3 channel BGR color image. 
    img = np.float32(img) / 255  # 转为float32类型,范围是[0,1]
    # Opencv loads as BGR:
    img = img[:, :, ::-1]  # BGR格式转换为RGB格式 shape: (224, 224, 3) 即(H, W, RGB)
    # input_img = preprocess_image(img)  # torch.Size([1, 3, 224, 224])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    input_img = photo
    grayscale_cam = grad_cam(input_img, target_category=None)  # shape: (300, 300)

    grayscale_cam_ = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))  
    cam = show_cam_on_image(img, grayscale_cam_)  # shape: (480, 640, 3) (480, 640)【0，1】
    cv2.imwrite("cam4cxq.jpg", cam)  # 保存图片 0-255  # (480, 640, 3)

    # # shape: (224, 224) # 这里要留意传入的形状是(w,h)  其实以上这行代码不需要执行,暂且先留着
    # grayscale_cam_ = Image.fromarray(np.uint8(grayscale_cam*255)) 
    # grayscale_cam_ = grayscale_cam_.resize((iw,ih),box=box)
    # grayscale_cam_ = (np.float32(grayscale_cam_) / 255)

    grayscale_cam_1 = Image.fromarray(np.uint8(grayscale_cam*255)) 
    # grayscale_cam_1.show()
    grayscale_cam_2 = grayscale_cam_1.resize((iw,ih),box=box)
    # grayscale_cam_2.show()
    grayscale_cam_3 = np.array(grayscale_cam_2, np.float32) / 255  # (480, 640)
    cam = show_cam_on_image(img, grayscale_cam_3) 
    cv2.imwrite("cam.jpg", cam)  # 保存图片 0-255  # (480, 640, 3)

    # cam = show_cam_on_image(img, grayscale_cam_)  # shape: (480, 640, 3) (480, 640)【0，1】
    # result4PIL = Image.fromarray(np.uint8(cam))  # size:
    # # result4PIL = result4PIL.resize((iw,ih),box=box)  # size:
    # result4PIL.save('cam.jpg')     
    
    # cv2.imwrite("cam.jpg陈旭旗", cam)  # 保存图片 0-255  # (480, 640, 3)
    
    
    ########################################################################################################

    # -----------------------------------------------------------------------------------













    # -----------------------------------------------------------------------------------
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # input_img.grad.zero_()  # AttributeError: 'NoneType' object has no attribute 'zero_'
    

    image = Image.open(args.image_path)
    image = image.convert('RGB')
    image_shape = np.array(np.shape(image)[0:2]) # 获得图片的尺寸
    crop_img = np.array(letterbox_image(image, (300,300)))  # (300, 300, 3)
    # photo = np.array(crop_img,dtype = np.float64) # 类型转为dtype = np.float64

    photo = torch.from_numpy(np.expand_dims(np.transpose(crop_img-MEANS,(2,0,1)),0))\
        .type(torch.FloatTensor).requires_grad_(True)  # 将颜色通道对应的维度调整到前面 torch.Size([1, 3, 300, 300])
    photo = photo.requires_grad_().cpu()  # .cpu()  # 范围是0-255


    gb = gb_model(photo, target_category=None)  # shape: (3, 300, 300) 相对于输入图像的梯度
    gb = gb.transpose((1, 2, 0))  # 调整通道在维度中的位置顺序 shape:(300, 300, 3)  相对于输入图像的梯度

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])  # shape:(224, 224, 3) # 由多个单通道的数组创建一个多通道的数组
    cam_gb = deprocess_image(cam_mask*gb)  # shape: (300, 300, 3) (300, 300, 3)
    result4PIL = Image.fromarray(np.uint8(cam_gb))  # size:(300, 300)
    result4PIL = result4PIL.resize((iw,ih),box=box)  # size:(640, 480)
    result4PIL.save('cam_gb.jpg')     
    
    
    # cv2.imwrite('cam_gb.jpg', cam_gb)  # 保存图片

    gb = deprocess_image(gb)  # shape: (300, 300, 3)
    result4PIL = Image.fromarray(np.uint8(gb))  # size:(300, 300)

    result4PIL = result4PIL.resize((iw,ih),box=box)  # size:(640, 480)
    result4PIL.save('gb.jpg')   
    # cv2.imwrite('gb.jpg', gb)  # 保存图片


    # -----------------------------------------------------------------------------------
    

    # cv2.imwrite("cam.jpg", cam)  # 保存图片
    # cv2.imwrite('gb.jpg', gb)  # 保存图片
    # cv2.imwrite('cam_gb.jpg', cam_gb)  # 保存图片

    # -----------------------------------------------------------------------------------


# 运行程序: python gradcam.py --image-path 1.jpg
# 运行程序: python gradcam.py --image-path ./examples/both.png


