# -*- coding: utf-8 -*- 
import numpy as np 
import math 
from .module import Module 
from ..functions import conv2d, conv_transpose2d 
from ...autograd import Variable 
 
class Conv1d(Module): 
    ''' 
    ''' 
    pass 
 
class Conv2d(Module): 
    '''Applies a 2D convolution over an input signal composed of several input planes.\n 
    Args: 
        in_channels (int): Number of channels in the input image 
        out_channels (int): Number of channels produced by the convolution 
        kernel_size (int|tuple of int): Size of the convolving kernel 
        stride (int|tuple of int): Stride of the convolution. Default: 1 
        padding (int|tuple of int):  Zero-padding added to both sides of the input. Default: 0 
        dilation (int|tuple of int): Spacing between kernel elements. Default: 1 
        bias (bool):  adds a learnable bias to the output. Default: True 
     
    Shape: 
        - Input: [N, in_channels, H, W] 
        - Output: [N, out_channels, H_out, W_out] 
 
        H_out = (H+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1 
        W_out = (W+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1 
 
    Reference: 
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md 
    ''' 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=True): 
        super().__init__() 
        if type(kernel_size) is int: 
            kernel_size = (kernel_size, kernel_size) 
        self.kernel = Variable(0.01*np.random.randn(out_channels, in_channels, *kernel_size)) 
        if bias: 
            self.bias = Variable(np.zeros((out_channels))) 
        else: 
            self.bias = None 
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation 
     
    def forward(self, x): 
        return conv2d(x, self.kernel, self.bias, self.stride, self.padding, self.dilation) 
 
class Conv3d(Module): 
    ''' 
    ''' 
    pass 

class ConvTranspose1d(Module): 
    ''' 
    ''' 
    pass 
 
class ConvTranspose2d(Module): 
    '''Applies a 2D transposed convolution operator over an input image composed of several input planes.\n 
    Args: 
        in_channels (int): Number of channels in the input image 
        out_channels (int): Number of channels produced by the convolution 
        kernel_size (int|tuple of int): Size of the convolving kernel 
        stride (int|tuple of int): Stride of the convolution. Default: 1 
        padding (int|tuple of int):  Zero-padding will be added to both sides of each dimension in the input. Default: 0 
        output_padding (int|tuple of int): Additional size added to one side of each dimension in the output shape. Default: 0 
        dilation (int|tuple of int): Spacing between kernel elements. Default: 1 
        bias (bool):  adds a learnable bias to the output. Default: True 
     
    Shape: 
        - Input: [N, in_channels, H, W] 
        - Output: [N, out_channels, H_out, W_out] 
 
        H_out = (H-1)*stride[0]-2*padding[0]+kernel_size[0]+output_padding[0] 
        W_out = (W-1)*stride[1]-2*padding[1]+kernel_size[1]+output_padding[1] 
    ''' 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, bias=True): 
        super().__init__() 
        if type(kernel_size) is int: 
            kernel_size = (kernel_size, kernel_size) 
        self.kernel = Variable(0.01*np.random.randn(out_channels, in_channels, *kernel_size)) 
        if bias: 
            self.bias = Variable(np.zeros((out_channels))) 
        else: 
            self.bias = None 
        self.stride = stride 
        self.padding = padding 
        self.output_padding = output_padding 
        self.dilation = dilation 
     
    def forward(self, x): 
        return conv_transpose2d(x, self.kernel, self.bias, self.stride, self.padding, self.output_padding, self.dilation)