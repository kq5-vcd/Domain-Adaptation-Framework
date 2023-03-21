import torch.nn as nn

def calculate_output_size(h_in, w_in, module):
    kernel = module.kernel_size
    if not isinstance(kernel, tuple):
        kernel = [kernel, kernel]
        
    stride = module.stride
    if not isinstance(stride, tuple):
        stride = [stride, stride]
        
    padding = module.padding
    if not isinstance(padding, tuple):
        padding = [padding, padding]
        
    dilation = module.dilation
    if not isinstance(dilation, tuple):
        dilation = [dilation, dilation]
    
    h_out = int((h_in + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1)
    w_out = int((w_in + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1)
    
    return h_out, w_out


def calculate_model_output_size(c_in, h_in, w_in, model):
    h = h_in
    w = w_in
    c = c_in
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for module in layer.children():
                if hasattr(module, 'kernel_size'):
                    h, w = calculate_output_size(h, w, module)
                    
                    if isinstance(module, nn.Conv2d):
                        c = module.out_channels
                    
        if hasattr(layer, 'kernel_size'):
            h, w = calculate_output_size(h, w, layer)
                    
            if isinstance(layer, nn.Conv2d):
                c = layer.out_channels
                
    return c, h, w