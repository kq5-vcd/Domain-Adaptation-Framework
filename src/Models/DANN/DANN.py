import torch.nn as nn

from src.Models.CNN.base_cnn import CNN
from src.Models.Classifiers.base_classifier import Classifier
from src.Models.DANN.ReverseLayer import ReverseLayerF
from src.Tools.model_utils import calculate_model_output_size

class DANN(nn.Module):    

    def __init__(self, c_in, h_in, w_in, output_size):
        super(DANN, self).__init__()
        
        self.in_c = c_in
        self.feature = CNN(c_in)
        
        c_out, h_out, w_out = calculate_model_output_size(c_in, h_in, w_in, self.feature)
        input_size = c_out * h_out * w_out
        
        self.class_classifier = Classifier(input_size, output_size)

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(input_size, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha=0):
        b, _, w, h = input_data.data.shape
        input_data = input_data.expand(b, self.in_c, w, h)
        
        feature = self.feature(input_data)
        feature = feature.view(b, -1)
        
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output