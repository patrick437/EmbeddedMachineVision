import math 
import torch
from utils.util import make_anchors

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k-1) + 1
    if P is None:
        p = k // 2
    return p


##Comnines conv and batchnorm layers
##This is done to reduce the number of layers in the model
def fuse_conv(conv, norm)  ##norm is the batch normalisation layer that follows immediately after the convolutional layer
    fused_conv = torch.nn.Conv2d(conv.in_channels, #number of input channels 
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

class conv(torch.nn.Module):

    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
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
    

class Residual(torch.nn.Module):
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
    


##backbone of the architecture brings together the 5 layers of the backbone togethere to form the DarkNet architecture
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

        self.p1 = torch.nn.Sequential(*p1)  ##Ensures data flows in the correct order throught the network
        self.p2 = torch.nn.Sequential(*p2)  ##makes the forward method alot tidier
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x): ##forward function of the DarkNet class
        p1 = self.p1(x)     ##passes each layer through the respective layers of the network
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5
    
##Feature pyramid network
class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2) ##upsample the feature map
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False) ##largest feature map
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = conv(width[3] + width[3], 3, 2) ##downsample the feature map
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False) ##medium feature map
        self.h5 = conv(width[4], width[4], 3, 2)  ##downsample the feature map
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False) ##smallest feature map


    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))  ## bridge in feature map upsampled to get h2 and concatenated to get h4
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))  ##largest feature map
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1)) ##medium feature map
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1)) ##smallest feature map
        return h2, h4, h6	
    
##Improves precision of bounding boxes treats the bounding box prediction as a regression problem
class DFL(torch.nn.Module):
    def __init__(self, ch=16):  ##ch is number of discrete values in each coordinates distribution
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.conv2d(ch, 1, 1, bias=False).requires_grad_(False) ##1x1 convolution which acts as a weighted sum (expected value calculator)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1) ##initialises weights to compute expected value distribution
        self.conv.weight.data[:] = torch.nn.Parameter(x)


    def forward(self, x):
        b, c, a = x.shape  ##b is batch size, c is number of classes, a is number of anchors
        #reshape to seperate the 4 coordinates (x, y, w, h) to each have there own ch distributuion 
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        ##apply softmax to convert the raw values to probability distributions, then calculate a expected value from this 
        ##result is converted to shape (b, 4, a) where b is batch size, 4 is the number of coordinates and a is the number of anchors
        return self.conv(x.softmax(1)).view(b, 4, a) 
    

class Head(torch.nn.Module):
    ##class variables of the Head class
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16
        self.nc = nc ##number of classes to detect
        self.nl = len(filters) ##number of detection layers
        self.no = nc + self.ch * 4 ##total outputs: classes + 4 coordinates with ditributions
        self.stride = torch.zeros(self.nl) ##stride for each detection layer

        ##dimenrsions for internal processing
        c1 = max(filters[0], self.nc) ##chaneel dimensions for classification branch
        c2 = max((filters[0] // 4, self.ch*4 )) ##channel dimensions for box branch

        ##Distribution focal loss module refined for coordinate prediction
        self.dfl = DFL(self.ch)

        ''' Classification branch one ofr each of the scales above (h2, h4, h6)
            Each branch: Conv(3x3) -> Conv(3x3) -> Conv(1x1) -> class prediction '''
        self.cls = torch.nn.ModuleList([torch.nn.Sequential(conv(x, c1, 3),
                                                            conv(c1, c1, 3),
                                                            torch.nn.Conv2d(c1, self.nc, 1)) for x in filters])  ##classification layer of prediction
        
        ''' Box branch one for each of the scales above (h2, h4, h6)
            Each branch: Conv(3x3) -> Conv(3x3) -> Conv(1x1) -> box prediction '''
        self.box = torch.nn.ModuleList([torch.nn.Sequential(conv(x, c1, 3),
                                                            conv(c2, c2, 3),
                                                            torch.nn.Conv2d(c2, 4*self.ch, 1)) for x in filters]) ##box layer of prediction
        

    def forward(self, x):

        ##forward pass of detection head

        ''' During training: returns raw predictions for loss calculation 
            During inference: processes predictions into final detections
            
            args: x (list): feature maps for each scale (h2, h4, h6) '''
        
        ##processes each feature map through box and class prediction layers and then concatenates the results
        for i in range(self.nl):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)

        ##if training return raw predictions
        if self.training:
            return x
        
        ##generates anchors and strides for each feature map
        self.anchors, self.strides = (x.totranspose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        ##reshape and concatenate predictions from all the feature maps
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x.split((self.ch*4, self.nc), 1) ##split componets into class and box components
        ##use dfl to do box predictions and split inot two parts
        a, b = torch.split(self.dfl(box), 2, 1)
        ##convert fro distance into actual coordinates
        a = self.anchors.unsqueeze(0) - a ##left top corner
        b = self.anchors.unsqueeze(0) + b ##right bottom
        box = torch.cat(((a+b)/2, b-a), 1)
        ##scale the box predictions and apply sigmoid to the class predictions
        return torch.cat((box * self.strides, cls.sigmoid()), 1) ##sigmoid converts form logits to probabilities(0-1)
    
    ##faster convergence, improves stability and reduces the risk of exploding gradients
    ##initialises the biases of the box and class prediction
    def iniationalize_biases(self):
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0  ##box initialisation
            #more classes lower intial confidence, 
            b[-1].bias.data[:m.nc] = math.log(5/m.nc / (640/s) **2) ##class bias initialisation

class YOLO(torch.nn.Module):
    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)

        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.iniationalize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))
    
    def fuse(self):
        for m in self.modules():
            if type(m) is conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
    
    ##can change size of model for different input sizes
    def yolo_v8_n(num_classes: int = 80):
        depth = [1, 2, 2]
        width = [3, 16, 32, 64, 128, 256]
        return YOLO(width, depth, num_classes)

        


        




