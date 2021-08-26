import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F
from torch.nn import Parameter


__all__ = ['ProHead', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class BasicBlock(nn.Module):
    expansion = 1


class Bottleneck():
    expansion = 4


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class ProHead(nn.Module):

    def __init__(self, block, num_classes=1000, zero_init_residual=False, two_branch=False, mlp=False, normlinear=False):
        super(ProHead, self).__init__()

        self.two_branch = two_branch
        self.mlp = mlp
        linear = NormedLinear if normlinear else nn.Linear

        if self.mlp:
            if self.two_branch:
                self.fc = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512 * block.expansion),
                    nn.ReLU()
                ) 
                self.instDis = linear(512 * block.expansion, num_classes)
                self.groupDis = linear(512 * block.expansion, num_classes)
            else:
                self.fc = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512 * block.expansion),
                    nn.ReLU(),
                    linear(512 * block.expansion, num_classes)
                ) 
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            if self.two_branch:
                self.groupDis = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        if self.mlp and self.two_branch:
            x = self.fc(x)
            x1 = self.instDis(x)
            x2 = self.groupDis(x)
            return [x1, x2]
        else:
            x1 = self.fc(x)
            if self.two_branch:
                x2 = self.groupDis(x)
                return [x1, x2]
            return x1


def _resnet(block, **kwargs):
    model = ProHead(block, **kwargs)
    return model


def resnet18(**kwargs):
    return _resnet(BasicBlock, **kwargs)


def resnet34(**kwargs):
    return _resnet(BasicBlock, **kwargs)


def resnet50(**kwargs):
    return _resnet(Bottleneck, **kwargs)


def resnet101(**kwargs):
    return _resnet(Bottleneck, **kwargs)


def resnet152(**kwargs):
    return _resnet(Bottleneck, **kwargs)

