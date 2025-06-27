import torch.nn as nn
import torch
from  SSSS import CBAM
import torch.nn as nn
import torch
from SSSS import CBAM


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ZFNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=False):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=1),#1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            CBAM(32),  # 添加CBAM
            BasicBlock(32, 32),  # 插入残差块
            nn.Conv2d(32, 64, kernel_size=5, stride=2,padding=1),#2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            CBAM(64),  # 添加CBAM
            BasicBlock(64, 64),  # 插入残差块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),#3
            nn.ReLU(inplace=True),
            CBAM(128),  # 添加CBAM
            BasicBlock(128, 128),  # 插入残差块
            nn.Conv2d(128, 128, kernel_size=3, padding=1),#4
            nn.ReLU(inplace=True),
            CBAM(128),  # 添加CBAM
            BasicBlock(128, 128),  # 插入残差块
            nn.Conv2d(128, 108, kernel_size=3, padding=1),#5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            CBAM(108),  # 添加CBAM
            BasicBlock(108, 108),  # 插入残差块
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(108 * 6 * 6, 2048),  # 注意这里的输入特征图大小是否匹配
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)