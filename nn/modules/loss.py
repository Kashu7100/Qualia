# -*- coding: utf-8 -*- 
import numpy as np 
import matplotlib.pyplot as plt 
from .module import Module 
from ...autograd import Variable, Backward 
 
class Regression(Module): 
    '''Base loss function class for Regression task\n 
    Regression is the task of approximating a mapping function (f) from input variables (x) to a continuous output variable (y). 
    A continuous output variable is a real-value, such as an integer or floating point value. These are often quantities, such as amounts and sizes. 
     
    Args: 
        live_plot (bool): if True, plot the loss as training of the model proceeds. 
    ''' 
    def __init__(self, live_plot=False): 
        self.live_plot = live_plot 
        self.losses = []
     
    def forward(self, x, t, *args): 
        raise NotImplementedError
     
    def _prepare_output(self, result, *args): 
        if not 'valid' in args and not 'test' in args: 
            self.losses.append(np.mean(result)) 
        if self.reduce: 
            if self.size_average: 
                result = Variable(np.mean(result)) 
            else: 
                result = Variable(np.sum(result)) 
        else: 
            result = Variable(result) 
        return result 
 
class Classification(Module): 
    '''Base loss function class for Classification task\n 
    Classification is the task of approximating a mapping function (f) from input variables (x) to discrete output variables (y). 
    The output variables are often called labels or categories. The mapping function predicts the class or category for a given observation. 
    ''' 
    def __init__(self, live_plot=False): 
        self.live_plot = live_plot 
        self.losses = []
     
    def forward(self, x, t, *args): 
        raise NotImplementedError
     
    def to_one_hot(self, x, classes): 
        ''' 
        Convert labels into one-hot representation 
         
        Args: 
            x (np.array): labels in shape of [N] 
            classes (int): number of classes to classify  
        ''' 
        labels = np.zeros((x.size, classes)) 
        for i, label in enumerate(labels): 
            label[x[i]] = 1 
        return labels 
     
    def get_acc(self, x, t): 
        if x.shape[1] != 1:
            pred = np.argmax(x.data, axis=1).reshape(-1,1) 
        else: 
            pred = x.data 
        if t.shape[1] != 1:
            label = np.argmax(t.data, axis=1).reshape(-1,1) 
        else: 
            label = t.data
        if pred.ndim != 2 or label.ndim != 2: 
            raise ValueError
        return np.sum(pred == label) / x.shape[0]

class MSELoss(Regression): 
    '''Mean Square Error, Quadratic loss, L2 Loss\n 
    Creates a criterion that measures the mean squared error between n elements in the input x and target t. 
     
    Args: 
        size_average (bool): the losses are averaged over observations for each minibatch. However, if False, the losses are instead summed for each minibatch. Ignored if reduce is False. 
        reduce (bool): the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch instead and ignores size_average. Default: True 
 
    Shape: 
        - Input: [N, C] 
        - Target: [N, C] 
        - Output: [1] by default 
                  [N] if not reduced 
    ''' 
    def __init__(self, size_average=True, reduce=True): 
        super().__init__() 
        self.size_average = size_average 
        self.reduce = reduce   
 
    def forward(self, x, t, *args): 
        if  x.shape != t.data.shape: 
           raise ValueError('[*] dimention of input {} and target {} did not match.'.format(x.shape, t.shape)) 
        result = np.sum(np.power(x.data - t.data,2),axis=1)/x.shape[1] 
        result = self._prepare_output(result, args)
        result.set_creator((MSELossBackward(result.shape, x, t)))
        return result

class MSELossBackward(Backward):
    def __init__(self, output_shape, var1, target):
        def f(x): 
            return 2*(var1.data - target.data)/var1.shape[0]  
        super().__init__(output_shape, var1, f) 
    
# TODO 
class HuberLoss(Regression): 
    '''Huber Loss, Smooth Mean Absolute Error\n 
    Huber loss is a loss function used in robust regression, that is less sensitive to outliers in data than the squared error loss.

    Args:
        delta (double): decide boundry value for Huber loss calculation. Default: 1
        size_average (bool): the losses are averaged over observations for each minibatch. However, if False, the losses are instead summed for each minibatch. Ignored if reduce is False. 
        reduce (bool): the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch instead and ignores size_average. Default: True 

    Shape:
        - Input: [N, C] 
        - Target: [N, C] 
        - Output: [1] by default 
                  [N] if not reduced 
    ''' 
    def __init__(self, delta=1, size_average=True, reduce=True): 
        super().__init__() 
        self.delta = delta
        self.size_average = size_average 
        self.reduce = reduce   
     
    def forward(self, x, t, *args): 
        if  x.shape != t.data.shape: 
            raise ValueError('[*] dimention of input and target did not match.') 
        a = x.data - t.data
        mask = (a <= self.delta)
        result = np.zeros(a.shape)
        result[mask] = (np.power(a,2)/2)[mask]
        result[np.invert(mask)] = (self.delta*(np.abs(a)-self.delta/2))[mask]
        result = self._prepare_output(result, args)
        result.set_creator((HuberBackward(result.shape, x, t, self.delta, mask)))
        return result
        
class HuberBackward(Backward):
    def __init__(self, output_shape, var1, target, delta, mask):
        def f(x):
            a = var1.data - target.data
            d = np.zeros(a.shape)
            d[mask] = a[mask]
            d[np.invert(mask)] = (delta*np.abs(a)/(a+1e-8))[mask]
            return d
        super().__init__(output_shape, var1, f)
 
# Classification 
class CrossEntropyLoss(Classification): 
    '''Cross Entropy Loss\n 
    It is useful when training a classification problem with C classes.
    This class incorporates the Softmax layer.  
 
    Args: 
        size_average (bool): the losses are averaged over observations for each minibatch. However, if False, the losses are instead summed for each minibatch. Ignored if reduce is False. 
        reduce (bool): the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch instead and ignores size_average. Default: True 
        live_plot (bool):  
 
    Shape: 
        - Input: [N,C] where C = number of classes 
        - Target: [N]  where each value is 0 ≤ targets[i] ≤ C-1 or 
                  [N,C] for one-hot representation 
        - Output: [1] as default 
                  [N] if reduce is False 
     
    Model: 
        L(p,q) = -sum(p(x)logq(x)) 
    ''' 
    def __init__(self, size_average=True, reduce=True, live_plot=False): 
        super().__init__(live_plot=live_plot) 
        self.size_average = size_average 
        self.reduce = reduce 
     
    def forward(self, x, t, *args): 
        if t.ndim is 1: 
            t.data = self.to_one_hot(t.data, x.shape[1]) 
        if  x.shape != t.shape:
            raise ValueError('[*] dimention of input {} and target {} did not match.'.format(x.shape, t.shape)) 
        c = np.max(x.data, axis=1) 
        c = np.expand_dims(c, axis=1)
        tmp = np.exp(x.data - c) 
        y = tmp / (np.expand_dims(np.sum(tmp, axis=1), axis=1) + 1e-8)
        result = np.sum(-t.data * np.log(y), axis=1)
        if not 'valid' in args and not 'test' in args: 
            self.losses.append(np.mean(result)) 
        if self.reduce: 
            if self.size_average: 
                result = Variable(np.mean(result)) 
            else: 
                result = Variable(np.sum(result)) 
        else: 
            result = Variable(result)
        result.set_creator((CrossEntropyLossBackward(result.shape, x, t))) 
        return result 
 
class CrossEntropyLossBackward(Backward): 
    def __init__(self, output_shape, var1, target): 
        def f(x):
            return (var1.data - target.data)/var1.shape[0] 
        super().__init__(output_shape, var1, f) 
 
class BCELoss(Classification): 
    '''Binary Cross Entropy Loss\n 
    This is used for measuring the error of a reconstruction in for example an auto-encoder.  
    Note that the targets y should be numbers between 0 and 1. 
     
    Args: 
        size_average (bool): the losses are averaged over observations for each minibatch. However, if False, the losses are instead summed for each minibatch. Ignored if reduce is False. 
        reduce (bool): the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch instead and ignores size_average. Default: True 
        live_plot (bool):  
 
    Shape: 
        - Input: [N,2] 
        - Target: [N]  where each value is 0 ≤ targets[i] ≤ 1 or 
                  [N,2] for one-hot representation 
        - Output: [1] as default 
                  [N] if not reduce is True 
     
    Model: 
        L(p,q) = -sum(p(x)logq(x)+(1-p(x))log(1-q(x))) 
    ''' 
    def __init__(self, size_average=True, reduce=True, live_plot=False): 
        super().__init__(live_plot=live_plot) 
        self.size_average = size_average 
        self.reduce = reduce 
     
    def forward(self, x, t, *args): 
        if t.ndim is 1: 
            t.data = self.to_one_hot(t.data, x.shape[1]) 
        if  x.shape != t.shape: 
            raise ValueError('[*] dimention of input and target did not match.') 
        c = np.max(x.data, axis=1) 
        c = np.expand_dims(c, axis=1) 
        tmp = np.exp(x.data - c) 
        y = tmp/np.expand_dims(np.sum(tmp, axis=1), axis=1) 
        result = np.sum(-t.data * np.log(y) - (1 - t.data) * np.log(1 - y), axis=1) 
        if not 'valid' in args and not 'test' in args: 
            self.losses.append(np.mean(result)) 
        if self.reduce: 
            if self.size_average: 
                result = Variable(np.mean(result)) 
            else: 
                result = Variable(np.sum(result)) 
        else: 
            result = Variable(result) 
        result.set_creator((BCELossBackward(result.shape, x, t))) 
        return result 
 
class BCELossBackward(Backward): 
    def __init__(self, output_shape, var1, target): 
        def f(x): 
            return (var1.data - target.data)/var1.shape[0]  
        super().__init__(output_shape, var1, f)
