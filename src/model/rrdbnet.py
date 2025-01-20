import torch
from torch import nn as nn
from torch.nn import functional as F


def make_layer(basic_block, num_basic_block, **kwargs):
    """
    Constructs a sequence of layers by stacking the same block multiple times.

    Args:
        basic_block (nn.Module): The block to be stacked.
        num_basic_block (int): The number of blocks to stack.
        **kwargs: Additional arguments to pass to the block.

    Returns:
        nn.Sequential: A sequential container of the stacked blocks.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwargs))
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block with 5 convolutions.
    This block is used to enhance feature extraction by densely connecting layers.
    """

    def __init__(self, nf=64, gc=32, bias=True):
        """
        Initializes the ResidualDenseBlock_5C.

        Args:
            nf (int): Number of input features. Default is 64.
            gc (int): Growth channel, i.e., intermediate channels. Default is 32.
            bias (bool): Whether to use bias in convolution layers. Default is True.
        """
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """
        Forward pass of the ResidualDenseBlock_5C.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the dense block operations.
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB).
    This block consists of multiple ResidualDenseBlock_5C layers.
    """

    def __init__(self, nf, gc=32):
        """
        Initializes the RRDB.

        Args:
            nf (int): Number of input features.
            gc (int): Growth channel, i.e., intermediate channels. Default is 32.
        """
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        """
        Forward pass of the RRDB.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the RRDB operations.
        """
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    Residual in Residual Dense Block Network (RRDBNet).
    This network is used for image super-resolution tasks.
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        """
        Initializes the RRDBNet.

        Args:
            in_nc (int): Number of input channels. Default is 3.
            out_nc (int): Number of output channels. Default is 3.
            nf (int): Number of features in the first convolution layer. Default is 64.
            nb (int): Number of RRDB blocks. Default is 23.
            gc (int): Growth channel, i.e., intermediate channels. Default is 32.
            scale (int): Scaling factor for super-resolution. Default is 4.
        """
        super(RRDBNet, self).__init__()
        self.scale = scale

        if scale == 2:
            in_nc = in_nc * 4
        elif scale == 1:
            in_nc = in_nc * 16

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = make_layer(RRDB, nb, nf=nf, gc=gc)
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # Upsampling layers
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """
        Forward pass of the RRDBNet.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the super-resolution operations.
        """
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        fea = self.conv_first(feat)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk

        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))

        return torch.clamp(out, 0, 1)


def pixel_unshuffle(x, scale):
    """
    Pixel unshuffle operation. This operation rearranges elements in the input tensor
    to reduce the spatial resolution and increase the number of channels.

    Args:
        x (Tensor): Input feature tensor with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: The pixel unshuffled feature tensor.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)