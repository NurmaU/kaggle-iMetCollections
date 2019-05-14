from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as M
import pretrainedmodels

ON_KAGGLE = False


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained: bool):
    if ON_KAGGLE and pretrained:
        net = net_cls()
        model_name = net_cls.__name__
        weights_path = f'../input/{model_name}/{model_name}.pth'
        net.load_state_dict(torch.load(weights_path))
    else:
        net = net_cls(pretrained=pretrained)
    return net


class ResNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.name = net_cls.__name__
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

        # self.net.fc.weight.data.normal_(0, 0.01)
        # self.net.fc.bias.data = -torch.tensor(1/num_classes).log()

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.densenet121):
        super().__init__()
        self.name = net_cls.__name__
        self.net = create_net(net_cls, pretrained=pretrained)
        self.avg_pool = AvgPool()
        self.net.classifier = nn.Linear(
            self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.net.classifier(out)
        return out

class PretrainedModels(nn.Module):
    def __init__(self, name, num_classes, pretrained=True):
        super().__init__()
        self.name = name
        self.net = pretrainedmodels.__dict__[self.name](num_classes=1000, pretrained='imagenet')
        self.net.last_linear = nn.Linear(self.net.last_linear.in_features, num_classes)
    
    def fresh_params(self):
        return self.net.last_linear.parameters()
    
    def forward(self, x):
        x = self.net(x)
        return x

resnet18 = partial(ResNet, net_cls=M.resnet18)
resnet34 = partial(ResNet, net_cls=M.resnet34)
resnet50 = partial(ResNet, net_cls=M.resnet50)
resnet101 = partial(ResNet, net_cls=M.resnet101)
resnet152 = partial(ResNet, net_cls=M.resnet152)

densenet121 = partial(DenseNet, net_cls=M.densenet121)
densenet169 = partial(DenseNet, net_cls=M.densenet169)
densenet201 = partial(DenseNet, net_cls=M.densenet201)
densenet161 = partial(DenseNet, net_cls=M.densenet161)

se_resnext101_32x4d = partial(PretrainedModels, name='se_resnext101_32x4d')
nasnetalarge = partial(PretrainedModels, name = 'nasnetalarge')
nasnetamobile = partial(PretrainedModels, name = 'nasnetamobile')
