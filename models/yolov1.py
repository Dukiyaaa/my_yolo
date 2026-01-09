import torch
import torchvision
from torch import nn
from torch.nn import Flatten, Sequential, Conv2d, MaxPool2d, Linear


class My_YOLO(nn.Module):
    def __init__(self):
        super(My_YOLO, self).__init__()
        self.model1 = Sequential(
            # 参数计算公式：
            # 卷积/池化:OH = (H + 2P - K) / S + 1
            # 输入：448x448x3 输出：224x224x64
            Conv2d(3, 64, 7, stride=2, padding=3),
            # 输入：224x224x64 输出：112x112x64
            MaxPool2d(2, 2),
        )
    def forward(self, x):
        x = self.model1(x)
        return x