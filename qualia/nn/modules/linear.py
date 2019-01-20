# -*- coding: utf-8 -*- 
import numpy as np 
import math 
from .module import Module 
from ..functions import linear, bilinear 
from ...autograd import Variable 
 
class Linear(Module): 
    '''Applies a linear transformation to the incoming data\n 
    Model: 
        y = x*w.T + b 
     
    Args: 
        in_features (int): size of each input sample 
        out_features (int): size of each output sample 
        bias (bool): whether to use bias. Default: True 
     
    Shape: 
        - Input: [N, *, in_features] where '∗' means any number of additional dimensions. 
        - Output: [N, *, out_features] where '∗' means any number of additional dimensions. 
    ''' 
    def __init__(self, in_features, out_features, bias=True): 
        super().__init__() 
        self.weight = Variable(np.random.normal(0, math.sqrt(1/in_features),(in_features, out_features))) 
        if bias: 
            self.bias = Variable(np.zeros(out_features)) 
        else: 
            self.bias = None 
     
    def forward(self, x): 
        return linear(x, self.weight, self.bias) 
 
class Bilinear(Module): 
    '''Applies a bilinear transformation to the incoming data\n 
    Model: 
        y = x1*A*x2 + b 
     
    Args: 
        in_features1 (int): size of each first input sample 
        in_features2 (int): size of each second input sample 
        out_features (int): size of each output sample 
        bias (bool): whether to use bias. Default: True 
     
    Shape: 
        - Input: [N, *, in_features1], [N, *, in_features2] where '∗' means any number of additional dimensions. 
        - Output: [N, *, out_features] where '∗' means any number of additional dimensions. 
    ''' 
    def __init__(self, in_features1, in_features2, out_features, bias=True): 
        super().__init__() 
        self.weight = Variable(np.random.normal(0, math.sqrt(1/(in_features1*in_features2)),(in_features1, in_features2, out_features))) 
        if bias: 
            self.bias = Variable(np.zeros(out_features)) 
        else: 
            self.bias = None 
     
    def forward(self, x1, x2): 
        return bilinear(x1, x2, self.weight, self.bias) 
