import math 
import torch
from utils.utils import make_anchors

##Comnines conv and batchnorm layers
##This is done to reduce the number of layers in the model
def fuse_conv(conv, norm)  ##norm is the batch normalisation layer that follows immediately after the convolutional layer
    fuse_conv = torch.nn.Conv2d(conv.in_channels, #number of input channels 
                                conv.out_channels, #number of output channels
                                conv.kernel_size,  #size of output channels
                                conv.stride,  ##step size of the convolutional
                                conv.padding,  ##padding size
                                bias=True).requires_grad_(False).to(conv.weight.device)
    
    ##The weights of the convolutional layer are multiplied by the weights of the batch normalisation layer
    w_conv = conv.weight.clone().detach().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.eps + norm.running_var))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(1) + b_norm)

    return fused_conv

class conv(torch.nn,module)
    def __init__(self, in_ch, out_ch, k=1, s-1, p=None, d=1, g=1):
        super(conv, self).__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k,p,d) d, g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu= torch.nn.SiLU(inplace=True)

    
    ##The forward function of the conv class
    ##This function applies the convolutional layer, batch normalisation and the activation function
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))
    
    ##The fuse_forward function of the conv class
    ##This function applies the convolutional layer
    def fuse_forward(self, x):
        return self.relu(self.conv(x))
    

class Residual(torch.nn.module):
    def __init__(self)

