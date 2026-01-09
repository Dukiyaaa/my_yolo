import torch
import torchvision
from torch import nn
from torch.nn import Flatten, Sequential, Conv2d, MaxPool2d, Linear, LeakyReLU


class My_YOLO(nn.Module):
    def __init__(self):
        super(My_YOLO, self).__init__()
        self.model1 = Sequential(
            # 参数计算公式：
            # 卷积/池化:OH = (H + 2P - K) / S + 1
            # 输入：448x448x3 输出：224x224x64
            Conv2d(3, 64, 7, stride=2, padding=3),
            # 根据论文，每个卷积层后跟一个leaky relu；加入激活函数的原因是将线性的卷积操作的结果转为非线性
            LeakyReLU(0.1),
            # 输入：224x224x64 输出：112x112x64
            MaxPool2d(2, 2),

            # 输入：112x112x64 输出：112x112x192 注意stride默认为1，padding默认为0
            Conv2d(64, 192, 3, stride=1, padding=1),
            LeakyReLU(0.1),
            # 输入：112x112x192 输出：56x56x192
            MaxPool2d(2, 2),

            # 输入：56x56x192 输出：56x56x128 注意padding默认为0
            Conv2d(192, 128, 1, stride=1, padding=0),
            LeakyReLU(0.1),
            # 输入：56x56x128 输出：56x56x256
            Conv2d(128, 256, 3, stride=1, padding=1),
            LeakyReLU(0.1),
            # 输入：56x56x256 输出：56x56x256
            Conv2d(256, 256, 1, stride=1, padding=0),
            LeakyReLU(0.1),
            # 输入：56x56x256 输出：56x56x512
            Conv2d(256, 512, 3, stride=1, padding=1),
            LeakyReLU(0.1),
            # 输入：56x56x512 输出：28x28x512
            MaxPool2d(2, 2),
            
            # 输入：28x28x512 输出：28x28x256 注意padding默认为0
            Conv2d(512, 256, 1, stride=1, padding=0),
            LeakyReLU(0.1),
            # 输入：28x28x256 输出：28x28x512
            Conv2d(256, 512, 3, stride=1, padding=1),
            LeakyReLU(0.1),
            # 输入：28x28x512 输出：28x28x256 注意padding默认为0
            Conv2d(512, 256, 1, stride=1, padding=0),
            LeakyReLU(0.1),
            # 输入：28x28x256 输出：28x28x512
            Conv2d(256, 512, 3, stride=1, padding=1),
            LeakyReLU(0.1),
            # 输入：28x28x512 输出：28x28x256 注意padding默认为0
            Conv2d(512, 256, 1, stride=1, padding=0),
            LeakyReLU(0.1),
            # 输入：28x28x256 输出：28x28x512
            Conv2d(256, 512, 3, stride=1, padding=1),
            LeakyReLU(0.1),
            # 输入：28x28x512 输出：28x28x256 注意padding默认为0
            Conv2d(512, 256, 1, stride=1, padding=0),
            LeakyReLU(0.1),
            # 输入：28x28x256 输出：28x28x512
            Conv2d(256, 512, 3, stride=1, padding=1),
            LeakyReLU(0.1),
            # 输入：28x28x512 输出：28x28x512
            Conv2d(512, 512, 1, stride=1, padding=0),
            LeakyReLU(0.1),
            # 输入：28x28x512 28x28x1024
            Conv2d(512, 1024, 3, stride=1, padding=1),
            LeakyReLU(0.1),
            # 输入：28x28x1024 输出：14x14x1024
            MaxPool2d(2, 2),

            # 输入：14x14x1024 输出：14x14x512
            Conv2d(1024, 512, 1, stride=1, padding=0),
            LeakyReLU(0.1),
            # 输入：14x14x512 输出：14x14x1024
            Conv2d(512, 1024, 3, stride=1, padding=1),
            LeakyReLU(0.1),
            # 输入：14x14x1024 输出：14x14x512
            Conv2d(1024, 512, 1, stride=1, padding=0),
            LeakyReLU(0.1),
            # 输入：14x14x512 输出：14x14x1024
            Conv2d(512, 1024, 3, stride=1, padding=1),
            LeakyReLU(0.1),
            # 输入：14x14x1024 输出：14x14x1024
            Conv2d(1024, 1024, 3, stride=1, padding=1),
            LeakyReLU(0.1),
            # 输入：14x14x1024 输出：7x7x1024
            Conv2d(1024, 1024, 3, stride=2, padding=1),
            LeakyReLU(0.1),

            # 输入：7x7x1024 输出：7x7x1024
            Conv2d(1024, 1024, 3, stride=1, padding=1),
            LeakyReLU(0.1),
            # 输入：7x7x1024 输出：7x7x1024
            Conv2d(1024, 1024, 3, stride=1, padding=1),
            LeakyReLU(0.1),

            Flatten(),
            Linear(7*7*1024, 4096),
            LeakyReLU(0.1),
            Linear(4096, 7*7*30),
        )

    def forward(self, x):
        x = self.model1(x)
        return x