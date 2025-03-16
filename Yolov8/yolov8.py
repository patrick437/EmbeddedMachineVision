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

class conv(torch.nn.Module)
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
    ##Internal skip connection inside the bottleneck of C2F block
    ##Used to keep local features in the network and big skip connections keep global features
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(conv(ch, ch, 3),
                                         conv(ch, ch, 3,))
        
    def forward(self, x):
        return x + self.res_m(x) if self.add_m else self.res_m(x)
    

##This is the bootleneck in the architecture and is carried out using two convolutionl layers and a residual connection initialised above
class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = conv(in_ch, out_ch//2)
        self.conv2 = conv(in_ch, out_ch//2)
        self.conv3 = conv((2+n)*out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList([Residual(out_ch//2, add) for _ in range(n)])

    def forward(self, x):
        y = self.conv1(x), self.conv2(x)
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, 1))
    
##Used to capture information at multiple layers and scales
class SPP(torch.nn.Module):
    ##Spatial pyramid pooling layer used to capture features at multiple scales
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = conv(in_ch, in_ch//2)  #resduce the channel by half
        self.conv2 = conv(in_ch * 2, out_ch) # processes the concatenated features
        self.res_m = torch.nn.MaxPool2d(k, 1, k//2) ##max pooling layer
        

    def forward(self, x):
        x = self.conv1(x)  ##reduce channels by in_chan//2
        y1 = self.res_m(x) ##apply max pooling once with kernel size k
        y2 = self.res_m(y1) ##apply max pooling agian with kernel size k
        return self.conv(torch.cat([x, y1, y2, self.res_m(y2)], 1)) ##concatenate the features and apply convolutional layer
    


class DarkNet(torch.nn.ModuleList):
    def __init__(self, width, depth):
        super().__init__()
        p1 = [conv(width[0], width[1], 3, 2)]
        p2 = [conv(width[1], width[2], 3, 2),
                CSP(width[2], width[2], depth[0])] ##CSP block
        p3 = [conv(width[2], width[3], 3, 2),
                CSP(width[3], width[3], depth[1])] ##CSP block
        p4 = [conv(width[3], width[4], 3, 2),
                CSP(width[4], width[4], depth[2])] ##CSP block
        p5 = [conv(width[4], width[5], 3, 2),
                CSP(width[5], width[5], depth[0]), ##CSP block
                SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5
    
class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = conv(width[3] + width[3], 3, 2)    
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)


    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))

        return h2, h4, h6	
    
##Improves precision of bounding boxes
class DFL(torch.nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.conv2d(ch, 1, 1, bias=False).requires_grad_(False) 
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)


    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)
    

class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16
        self.nc = nc
        self.nl = len(filters)
        self.no = nc + self.ch * 4
        self.stride = torch.zeros(self.nl)

        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch*4 ))

        self.dfl = DFL(self.ch)
        self.cls = torch.nn.ModuleList([torch.nn.Sequential(conv(x, c1, 3),
                                                            conv(c1, c1, 3),
                                                            torch.nn.Conv2d(c1, self.nc, 1)) for x in filters])  ##classification layer of prediction
        self.box = torch.nn.ModuleList([torch.nn.Sequential(conv(x, c1, 3),
                                                            conv(c2, c2, 3),
                                                            torch.nn.Conv2d(c2, 4*self.ch, 1)) for x in filters]) ##box layer of prediction
        

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)

        if self.training:
            return x
        
        self.anchors, self.strides = (x.totranspose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x.split((self.ch*4, self.nc), 1)
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a+b)/2, b-a), 1)
        return torch.cat((box * self.strides, cls.sigmoid()), 1)
    
    def iniationalize_biases(self):

        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:m.nc] = math.log(5/m.nc / (640/s) **2)



        




