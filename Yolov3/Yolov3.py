import torch
import torch.nn as nn

config = [
    # Darknet-53 backbone
    # First block
    (32, 3, 1),  # (out_channels, kernel_size, stride)
    (64, 3, 2),  # Downsample
    ["R", 1],    # Residual block with 1 repeat
    
    # Second block
    (128, 3, 2),  # Downsample
    ["R", 2],     # Residual block with 2 repeats
    
    # Third block
    (256, 3, 2),  # Downsample
    ["R", 8],     # Residual block with 8 repeats
    
    # Fourth block
    (512, 3, 2),  # Downsample
    ["R", 8],     # Residual block with 8 repeats
    
    # Fifth block
    (1024, 3, 2),  # Downsample
    ["R", 4],      # Residual block with 4 repeats
    
    # YOLOv3 specific layers
    # First detection branch - Large objects
    (512, 1, 1),
    (1024, 3, 1),
    (512, 1, 1),
    (1024, 3, 1),
    (512, 1, 1),
    (1024, 3, 1),
    "S",  # Scale prediction for large objects
    
    # Upsample and merge with earlier feature map
    (256, 1, 1),
    "U",  # Upsample
    
    # Second detection branch - Medium objects
    (256, 1, 1),
    (512, 3, 1),
    (256, 1, 1),
    (512, 3, 1),
    (256, 1, 1),
    (512, 3, 1),
    "S",  # Scale prediction for medium objects
    
    # Upsample and merge with earliest feature map
    (128, 1, 1),
    "U",  # Upsample
    
    # Third detection branch - Small objects
    (128, 1, 1),
    (256, 3, 1),
    (128, 1, 1),
    (256, 3, 1),
    (128, 1, 1),
    (256, 3, 1),
    "S",  # Scale prediction for small objects
]
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
        super().__init__()
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
            residual = x
            x = layer(x)  
            if self.use_residual:
                x = x + residual 
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels*2, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, (num_classes+5)*3, bn_act=False, kernel_size=1)  ##For every cell we have threee anchor boxes
        )
        self.num_classes = num_classes

    def forward(self, x):
        return(
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes+5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
    

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = [] ##store to concatenate the layers

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs


    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels, 
                        out_channels, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=1 if kernel_size == 3 else 0
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
                    
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3 ## When we upsample we also want to concatenate

        return layers      


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes+5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes+5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes+5)
    print("Success") 