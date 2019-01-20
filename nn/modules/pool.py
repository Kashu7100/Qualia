# -*- coding: utf-8 -*- 
from .module import Module 
from ..functions import max_pool2d 
 
class MaxPool1d(Module): 
    ''' 
    ''' 
    pass 
 
class MaxPool2d(Module): 
    '''Applies a 2D max pooling over an input signal composed of several input planes.\n 
    Args: 
        kernel_size (tuple of int): the size of the window to take a max over 
        stride (int|tuple): the stride of the window. Default value is kernel_size 
        padding (int|tuple): implicit zero padding to be added on both sides 
        dilation (int|tuple): a parameter that controls the stride of elements in the window 
 
    Shape: 
        - Input: [N, channels, H, W] 
        - Output: [N, channels, H_out, W_out] 
 
        H_out = (H+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1 
        W_out = (W+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1 
    ''' 
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1): 
        super().__init__() 
        self.kernel_size = kernel_size 
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation 
     
    def forward(self, x): 
        return max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation) 
     
class AvePool1d(Module): 
    ''' 
    ''' 
    pass 
 
class AvePool2d(Module): 
    ''' 
    ''' 
    pass