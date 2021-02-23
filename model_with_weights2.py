"""
Implementation of Yolo (v3) architecture

paper (it's srsly hilarious):

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ScalePrediction, self).__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []
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
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels += in_channels * 2

        return layers

    def load_CNN_weights(self, ptr, block):

        conv_layer = block.conv
        if block.use_bn_act:
            # Load BN bias, weights, running mean and running variance
            bn_layer = block.bn
            num_b = bn_layer.bias.numel()  # Number of biases
            # Bias
            bn_b = torch.from_numpy(self.weights[ptr : ptr + num_b]).view_as(
                bn_layer.bias
            )
            bn_layer.bias.data.copy_(bn_b)
            ptr += num_b
            # Weight
            bn_w = torch.from_numpy(self.weights[ptr : ptr + num_b]).view_as(
                bn_layer.weight
            )
            bn_layer.weight.data.copy_(bn_w)
            ptr += num_b
            # Running Mean
            bn_rm = torch.from_numpy(self.weights[ptr : ptr + num_b]).view_as(
                bn_layer.running_mean
            )
            bn_layer.running_mean.data.copy_(bn_rm)
            ptr += num_b
            # Running Var
            bn_rv = torch.from_numpy(self.weights[ptr : ptr + num_b]).view_as(
                bn_layer.running_var
            )
            bn_layer.running_var.data.copy_(bn_rv)
            ptr += num_b
        else:
            # Load conv. bias
            num_b = conv_layer.bias.numel()

            conv_b = torch.from_numpy(self.weights[ptr : ptr + num_b]).view_as(
                conv_layer.bias
            )
            conv_layer.bias.data.copy_(conv_b)
            ptr += num_b
            # Load conv. weights
        num_w = conv_layer.weight.numel()
        conv_w = torch.from_numpy(self.weights[ptr : ptr + num_w]).view_as(
            conv_layer.weight
        )
        conv_layer.weight.data.copy_(conv_w)
        ptr += num_w
        return ptr

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(
                f, dtype=np.int32, count=5
            )  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            self.weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, CNNBlock):
                ptr = self.load_CNN_weights(ptr, layer)

            elif isinstance(layer, ResidualBlock):
                for i in range(layer.num_repeats):
                    ptr = self.load_CNN_weights(ptr, layer.layers[i][0])
                    ptr = self.load_CNN_weights(ptr, layer.layers[i][1])

            elif isinstance(layer, ScalePrediction):
                # print("Starting scale prediction route")
                cnn_block = layer.pred[0]
                last_block = layer.pred[1]
                ptr = self.load_CNN_weights(ptr, cnn_block)
                ptr = self.load_CNN_weights(ptr, last_block)

                # ptr = self.load_CNN_weights(ptr, cnn_block)
            # print("Scale prediction ")

        print(ptr)


if __name__ == "__main__":

    model = YOLOv3()
    model.load_darknet_weights(weights_path="yolov31.weights")
    model.layers[15].pred[1] = CNNBlock(1024, 25 * 3, bn_act=False, kernel_size=1)
    model.layers[22].pred[1] = CNNBlock(512, 25 * 3, bn_act=False, kernel_size=1)
    model.layers[29].pred[1] = CNNBlock(256, 25 * 3, bn_act=False, kernel_size=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    from utils import save_checkpoint

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    import sys

    sys.exit()
