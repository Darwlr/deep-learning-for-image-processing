import torch.nn as nn
import torch

class AlexNet(nn.Module):
    """
    num_classes: 最后的分类数
    init_weights：是否需要进行权重的初始化
    """
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 卷积层
        self.features = nn.Sequential(
            # layer one
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), # input:[3, 224, 224] output:[96, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output:[96, 27, 27]
            # layer two
            nn.Conv2d(96, 256, kernel_size=5, padding=2), # output: [256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # output: [256, 13, 13]
            # layer three
            nn.Conv2d(256, 384, kernel_size=3, padding=1), # output: [384, 13, 13]
            nn.ReLU(inplace=True),
            # layer four
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # output: [384, 13, 13]
            nn.ReLU(inplace=True),
            # layer five
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # output: [256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # output: [256, 6, 6]
        )
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # 随机失活率：0.5
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initial_weights()
    # 前向传播
    def forward(self, x):
        x = self.features(x)
        # x原本是一个四维向量:[batches, channel, height, width],
        # 将向量x转成一维的，不包括第0个维度
        # 相当于：channel * width * height
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    # 初始化权重
    def _initial_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

