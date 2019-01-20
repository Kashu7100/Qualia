# -*- coding: utf-8 -*- 
from .module import Module 
from ..functions import normalize, reshape 
from ...autograd import Variable 
import numpy as np
 
class BatchNorm1d(Module): 
    ''' 
    ''' 
    pass 
 
class BatchNorm2d(Module): 
    '''Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)\n 
    Because the Batch Normalization is done over the C dimension, computing statistics on (N, H, W) slices,  
    it’s common terminology to call this Spatial Batch Normalization. 
    Args: 
        num_features (int): C from an expected input of size (N,C,H,W) 
        affine (bool): a boolean value that when set to True, this module has learnable affine parameters. Default: True 
        eps (float): a value added to the denominator for numerical stability. Default: 1e-10 
     
    Shape: 
        - Input: [N, C, H, W] 
        - Output: [N, C, H, W] 
    ''' 
    def __init__(self, num_features, affine=True, eps=1e-10): 
        super().__init__() 
        if affine: 
            self.weight = Variable(np.random.rand(num_features))  
            self.bias = Variable(np.zeros(num_features)) 
        else: 
            self.weight = None 
            self.bias = None 
        self.eps = eps 
     
    def forward(self, x): 
        if self.weight is None and self.bias is None: 
            return normalize(x, axis=1, eps=self.eps) 
        else: 
            return normalize(x, axis=1, eps=self.eps)*reshape(self.weight, (1,-1,1,1)) + reshape(self.bias, (1,-1,1,1))