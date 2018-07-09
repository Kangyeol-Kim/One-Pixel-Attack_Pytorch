import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import argparse
import os
import sys

import time
import datetime

from utils import *

sys.path.insert(0, './networks')
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vgg16', help='vgg16 | network_in_network')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--n_epoches', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--use_cuda', type=str2bool, default='True')
parser.add_argument('--use_nsml', type=str2bool, default='True')

parser.add_argument("--pause", type=int, default=0)
parser.add_argument('--mode', type=str, default='train')

config = parser.parse_args()
if config.use_nsml:
    import nsml
    from nsml import DATASET_PATH
    from nsml import GPU_NUM
    config.use_cuda = GPU_NUM

if config.use_cuda:
    use_cuda = torch.cuda.is_available()


## Loadin CIFAR10

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# path = os.path.join(DATASET_PATH, 'train')
# trainset = ImageFolder(root=path, transform=transform)
trainset = torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# import net
if config.model == 'vgg16':
    from vgg16 import *
    net = VGG16('VGG16')
elif config.model == 'network_in_network':
    from network_in_network import *
    net = NiN('NiN')

if config.use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = config.lr)

start_time = time.time()
loader_iter = iter(loader)

for epoch in range(config.n_epoches):
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if config.use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outs = net(inputs)
        loss = criterion(outs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    elapsed = time.time() - start_time
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Elapsed: {}, Epoch: [{}/{}], Loss: {}"
             .format(elapsed, (epoch+1), config.n_epoches, loss.data))


torch.save(net, config.model + 'pt')
