# -*- coding: utf-8 -*- 
from .module import Module 
from ..functions import sin, tanh, relu, leakyrelu, elu, sigmoid 

'''
##############################################################
Use of activation functions not classes are highly recommended
##############################################################
'''
 
class Sin(Module): 
    '''Trigonometric sine, element-wise.\n 
    ''' 
    def __init__(self): 
        super().__init__()     
         
    def forward(self, x): 
        return sin(x) 
 
class Tanh(Module): 
    '''Compute hyperbolic tangent element-wise.\n 
    ''' 
    def __init__(self): 
        super().__init__()     
         
    def forward(self, x): 
        return tanh(x) 
 
class ReLU(Module): 
    '''Rectified Linear Unit\n 
    ''' 
    def __init__(self): 
        super().__init__()     
         
    def forward(self, x): 
        return relu(x) 
 
class LeakyReLU(Module): 
    '''Leaky Rectified Linear Unit     
    ''' 
    def __init__(self): 
        super().__init__() 
 
    def forward(self, x): 
        return leakyrelu(x) 
 
class ELU(Module): 
    '''Exponential Linear Unit\n 
 
    Args: 
        a (float): a positive parameter that scales the negative region of the input so that the output region will be [-a, +inf] 
    ''' 
    def __init__(self, a): 
        super().__init__() 
        self.a = a     
         
    def forward(self, x): 
        return elu(x, self.a) 
 
class Sigmoid(Module): 
    '''A logistic function\n 
    ''' 
    def __init__(self): 
        super().__init__()     
         
    def forward(self, x): 
        return sigmoid(x) 
