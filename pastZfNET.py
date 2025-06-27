import torch.nn as nn
import torch


class ZFNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=False):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(  # 打包
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # input[3, 224, 224]  output[48, 110, 110] 自动舍去小数点后
            nn.ReLU(inplace=True),  # inplace 可以载入更大模型
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       # output[48, 55, 55] kernel_num为原论文一半  1
            nn.Conv2d(32, 64, kernel_size=5, stride=2),            # output[128, 26, 26]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       # output[128, 13, 13]    2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),                                                         #3
            nn.Conv2d(128, 128, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),                                                        #4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]    #5
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # 全连接
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),#6
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 展平   或者view()
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
                nn.init.constant_(m.bias, 0)

