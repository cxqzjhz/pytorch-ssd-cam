from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable
from ssd import SSD

ssd = SSD()
model = ssd.net.module
writer = SummaryWriter()
for i in range(5):
    images = torch.randn(4, 3, 300, 300).cuda()
    writer.add_graph(model, input_to_model=images, verbose=False)
writer.flush()
writer.close()