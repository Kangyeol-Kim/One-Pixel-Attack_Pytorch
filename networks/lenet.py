import torch
import torch.nn as nn

#  refer https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220648539191&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F

LeNet_config = [6, 'AP', 16, 'AP', 120]

class LeNet(nn.Module):
    def __init__(self, name):
        super(LeNet, self).__init__()
        self.layers = self.make_layers(name)
        self.fc = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, 10)

    def forward(self, x):
        out = self.layers(x)
        out = out.squeeze()
        out = self.fc(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def make_layers(self, LeNet_config):
        layers = []
        input_depth = 3
        for l in LeNet_config:
            if l == 'AP':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(input_depth, l, kernel_size=5, stride=1)]
                input_depth = l
        return nn.Sequential(*layers)


# test code
net = LeNet(LeNet_config)
x = torch.randn(2, 3, 32, 32)
y = net(x)
print(y.size())
