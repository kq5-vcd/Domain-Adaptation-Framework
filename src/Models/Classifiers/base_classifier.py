import torch.nn as nn

class Classifier(nn.Module):    

    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(input_size, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, output_size))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())
        
    def forward(self, x):
        return self.class_classifier(x)