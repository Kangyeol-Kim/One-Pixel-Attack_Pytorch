# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
import torch
import torch.nn as nn

cfg = { 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

class VGG16(nn.Module):
    def __init__(self, net_name):
        super(VGG16, self).__init__()
        self.layers = self.make_layers(cfg[net_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def make_layers(self, layer_config):
        layers = []
        input_depth = 3
        for l in layer_config:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(input_depth, l, kernel_size=3, padding=1),
                           nn.BatchNorm2d(l), # output size indicate
                           nn.ReLU(inplace=True)]
                input_depth = l
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
