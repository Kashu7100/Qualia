# -*- coding: utf-8 -*- 
import numpy as np 
 
class Optimizer(object): 
    '''Optimizer base class\n 
    ''' 
    def __init__(self, parameters, lr): 
        self.params = parameters 
        self.lr = lr 
     
    def step(self): 
        raise NotImplementedError 
 
class SGD(Optimizer): 
    '''Stochastic Gradient Descent\n 
     
    Args: 
        parameters (OrderedDict): parameters stored in a Module 
        lr (float): learning rate 
 
    Attributes: 
        params (generator): generator named params in Module class 
        lr (float): learning rate 
    ''' 
    def __init__(self, parameters, lr=0.001): 
        super().__init__(parameters, lr) 
         
    def step(self): 
        for i in self.params(): 
            i.data -= self.lr * i.grad 
 
class Momentum(Optimizer): 
    '''Momentum SGD\n 
 
    Args: 
        parameters (OrderedDict): parameters stored in a Module 
        lr (float): learning rate 
        momentum (float): momentum 
 
    Attributes: 
        params (generator): generator named params in Module class 
        lr (float): learning rate 
        momentum (float): momentum 
        velocity (dict): stores velocities for each parameter 
    ''' 
    def __init__(self, parameters, lr=0.001, momentum=0.9): 
        super().__init__(parameters, lr) 
        self.momentum = momentum 
        self.velocity = {} 
     
    def step(self): 
        for i, var in enumerate(self.params()): 
            if i not in self.velocity:  
                self.velocity[i] = np.zeros_like(var.grad) 
            self.velocity[i] = self.momentum * self.velocity[i] + (1 - self.momentum) * var.grad 
            var.data -= self.lr * self.velocity[i] 
 
class AdaGrad(Optimizer): 
    '''Adaptive Subgradient\n 
    Adagrad is an optimizer with parameter-specific learning rates, which are adapted relative to how frequently a parameter gets updated during training.  
    The more updates a parameter receives, the smaller the updates. 
 
    Args: 
        parameters (OrderedDict): parameters stored in a Module 
        lr (float): learning rate 
        eps (float): constant that stablizes the calculation 
 
    Attributes: 
        params (generator): generator named params in Module class 
        lr (float): learning rate 
        eps (float): constant 
        h (dict): stores adoptive term 
    ''' 
    def __init__(self, parameters, lr=0.001, eps=1e-8): 
        super().__init__(parameters, lr) 
        self.eps = eps 
        self.h = {} 
     
    def step(self): 
        for i, var in enumerate(self.params()): 
            if i not in self.h:  
                self.h[i] = np.zeros_like(var.grad) 
            self.h[i] += var.grad * var.grad 
            var.data -= self.lr * var.grad / np.sqrt(self.h[i]+self.eps)  
 
class Adadelta(Optimizer): 
    '''ADADELTA\n 
    This method dynamically adapts over time using only first order information and has minimal computational overhead beyond vanilla stochastic gradient descent. 
     
    Args: 
        parameters (OrderedDict): parameters stored in a Module 
        lr (float): learning rate 
        decay_rate (float): decay rate 
        eps (float): constant that stablizes the calculation 
 
    Attributes: 
        params (generator): generator named params in Module class 
        gamma (float): decay rate 
        eps (float): constant 
        g (dict): accumulate grads 
        u (dict): accumulate updates 
 
    Reference: 
        https://arxiv.org/pdf/1212.5701.pdf 
    ''' 
    def __init__(self, parameters, decay_rate=0.95, eps=1e-6): 
        super().__init__(parameters, None) 
        self.rho = decay_rate 
        self.eps = eps 
        self.g = {} 
        self.u = {} 
     
    def step(self): 
        for i, var in enumerate(self.params()): 
            if i not in self.g:  
                self.g[i] = np.zeros_like(var.grad) 
            if i not in self.u: 
                self.u[i] = np.zeros_like(var.grad) 
            self.g[i] = self.rho * self.g[i] + (1-self.rho) * var.grad**2 
            update = -np.sqrt(self.u[i]+self.eps) * var.grad / np.sqrt(self.g[i]+self.eps) 
            self.u[i] = self.rho * self.u[i] + (1-self.rho) * update**2 
            var.data += update 
 
class RMSProp(Optimizer): 
    '''RMSProp\n 
    This optimizer is usually a good choice for recurrent neural networks. 
 
    Args: 
        parameters (OrderedDict): parameters stored in a Module 
        lr (float): learning rate 
        decay_rate (float): forgets past gradients at rate of this 
        eps (float): 
 
    Attributes: 
        params (generator): generator named params in Module class 
        lr (float): learning rate 
        gamma (float): forgets past gradients at rate of this 
        eps (float): 
        h (dict): stores adoptive term 
    ''' 
    def __init__(self, parameters, lr=0.001, decay_rate=0.99, eps=1e-8): 
        super().__init__(parameters, lr) 
        self.gamma = decay_rate 
        self.eps = eps 
        self.h = {} 
     
    def step(self): 
        for i, var in enumerate(self.params()): 
            if i not in self.h:  
                self.h[i] = np.zeros_like(var.grad) 
            self.h[i] = self.gamma * self.h[i] + (1-self.gamma) * var.grad**2 
            var.data -= self.lr * var.grad / np.sqrt(self.h[i]+self.eps) 
