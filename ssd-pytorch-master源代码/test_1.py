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
    # flt = output.contiguous().view(num, -1, 5)  # 这几行代码注释掉之后程序仍然能够正确运行
    # _, idx = flt[:, :, 0].sort(1, descending=True)  # 这几行代码注释掉之后程序仍然能够正确运行
    # _, rank = idx.sort(1)  # 这几行代码注释掉之后程序仍然能够正确运行
    # flt[(rank < 200).unsqueeze(-1).expand_as(flt)].fill_(0)  # 这几行代码注释掉之后程序仍然能够正确运行  
    # # 注意这里的操作并不会影响output,因为flt[mask].fill_(0)不会影响output
    return output  # torch.Size([1, 3, 200, 5])  1置信度+4位置信息
    
    



if __name__ == '__main__':
    import torch
    torch.manual_seed(seed=20200910)
    loc_data = torch.randn(1, 8732, 4).requires_grad_()
    conf_data = torch.randn(1, 8732, 3).requires_grad_()
    prior_data = torch.randn(8732, 4).requires_grad_()

    result = detect_forward(loc_data, conf_data, prior_data)
    torch.sum(result).backward()


