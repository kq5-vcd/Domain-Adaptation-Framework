import torch.nn as nn

class CNN(nn.Module):    

    def __init__(self, in_c):
        super(CNN, self).__init__()
        
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(in_c, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        
    def forward(self, x):
        return self.feature(x)