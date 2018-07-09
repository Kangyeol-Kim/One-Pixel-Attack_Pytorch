'''
cifar-10 version
ImageNet should be added
'''
import torch
import torch.nn as nn

cfg = {'NiN': [192, 160, 96, 'M', 'D', 192, 192, 192, 'A', 'D', '192-1', '192-2', 10]}

class NiN(nn.Module):
    def __init__(self, net_name):
        super(NiN, self).__init__()
        self.layers = self.make_layers(cfg[net_name])
        nn.classifier = nn.Linear(640, 10)
        self.softmax = nn.Softmax()

    def forward(self, input):
        out = self.layers(input)
        out = out.view(out.size(0), -1) # flatten
        out = nn.classifier(out)
        out = self.softmax(out)
        return out

    def make_layers(self, layer_config):
        layers = []
        input_dim = 3
        for layer in layer_config:
            if layer == 'M':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            elif layer == 'A':
                layers.append(nn.AvgPool2d(kernel_size=3, stride=2))
            elif layer == 'D':
                layers.append(nn.Dropout(p=0.5))
            elif layer == 192:
                layers += [nn.Conv2d(input_dim, layer, kernel_size=5, stride=1, padding=1),
                           nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]]
                input_dim = layer
            elif layer == '192-1':
                layers += [nn.Conv2d(input_dim, 192, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(192), nn.ReLU(inplace=True)]
                input_dim = 192
            elif layer == '192-2':
                layers += [nn.Conv2d(input_dim, 192, kernel_size=1, stride=1, padding=1),
                           nn.BatchNorm2d(192), nn.ReLU(inplace=True)]
                input_dim = 192
            else:
                layers += [nn.Conv2d(input_dim, layer, kernel_size=1, stride=1, padding=1),
                           nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
                input_dim = layer

        return nn.Sequential(*layers)

# xx = torch.randn((64, 3, 32, 32))
# net = NiN('NiN')
# net(xx).size()
