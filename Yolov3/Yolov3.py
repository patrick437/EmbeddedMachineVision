import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:   ##Might lose some performance with if statement
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super.__init__()
        self.layers = nn.ModuleList()  ##going to have alot of repeats
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1) ##downsamples the numbr of filters and brings it back again.
                )
            ]
        self.use_residual = use_residual  
        self.num_repeats = num_repeats


    def forward(self, x):

        for layer in self.layers:
            if self.use_residual:
                return x + self.layers(x)
            else:
                return self.layers(x)
            
        return x
        

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels*2, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, (num_classes+5)*3, bn_act=False, kernel_size=1)  ##For every cell we have threee anchor boxes
            """ for every cell we have three anchor boxes, each anchor box has 5+num_classes values
              each class output [po,x,y,w,h] therefore we need 5 valuse for each class"""
        )
        self.num_classes = num_classes

    def forward(self, x):
        return(
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes+5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

class Yolov3(nn.Module):
    pass