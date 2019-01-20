# -*- coding: utf-8 -*- 
from collections import OrderedDict 
import numpy as np 
import math 
import h5py as h5 
from ...autograd import Variable 
 
class Module(object): 
    '''Base class for all neural network modules in qualia.\n 
    Module can incoporate Modules, allowing to nest them in a tree structure. 
    
    Examples::
        >>> class DeepConvNet(Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         # (N,1,28,28) -> (N,16,28,28)
        >>>         self.conv1 = Conv2d(1, 16, 5, padding=2)
        >>>         # (N,16,28,28) -> (N,16,14,14)
        >>>         self.pool1 = MaxPool2d((2,2))
        >>>         # (N,16,14,14) -> (N,16,14,14)
        >>>         self.conv2 = Conv2d(16, 16, 5, padding=2)
        >>>         # (N,16,14,14) -> (N,16,7,7)
        >>>         self.pool2 = MaxPool2d((2,2))
        >>>         # Reshape
        >>>         self.linear1 = Linear(16*7*7, 128)
        >>>         self.linear2 = Linear(128, 10)
        >>>
        >>>     def forward(self, x):
        >>>         x = leakyrelu(self.conv1(x))
        >>>         x = self.pool1(x)
        >>>         x = leakyrelu(self.conv2(x))
        >>>         x = self.pool2(x)
        >>>         x = reshape(x, (-1, 16*7*7))
        >>>         x = leakyrelu(self.linear1(x))
        >>>         x = leakyrelu(self.linear2(x))
        >>>         return x
    ''' 
    def __init__(self): 
        self._modules = OrderedDict() 
        self._params = OrderedDict() 
        self.training = True 
 
    def __setattr__(self, key, value): 
        if isinstance(value, Module):
            self._modules[key] = value 
        elif isinstance(value, Variable):
            self._params[key] = value 
        else:
            object.__setattr__(self, key, value) 
     
    def __getattr__(self, key): 
        if self._modules:
            return self._modules[key]
        elif self._params:
            return self._params[key]
        else:
            object.__getattr__(self, key)
     
    def __call__(self, *args, **kwargs): 
        return self.forward(*args, **kwargs) 
     
    def forward(self, *args, **kwargs): 
        raise NotImplementedError 
 
    def params(self): 
        if not self._modules: 
            for _, var in self._params.items(): 
                yield var 
        else:
            for _, module in self._modules.items(): 
                for _, var in module._params.items(): 
                    yield var 
     
    def zero_grad(self): 
        if not self._modules: 
            for _, var in self._params.items(): 
                var.grad = None 
        else: 
            for _, module in self._modules.items(): 
                for _, var in module._params.items(): 
                    var.grad = None 
     
    def save(self, filename): 
        '''Saves internal parameters of the Module in HDF5 format.\n 
        Args: 
            filename (str): specify the filename as well as the saving path without the file extension. (ex) path/to/filename 
        ''' 
        with h5.File(filename + '.hdf5', 'w') as file: 
            if not self._modules: 
                for key, value in self._params.items(): 
                    file.create_dataset(str(key), dtype='f8', data=value.data) 
            else: 
                for name, module in self._modules.items(): 
                    grp = file.create_group(str(name)) 
                    for key, value in module._params.items(): 
                        grp.create_dataset(str(key), dtype='f8', data=value.data) 
     
    def load(self, filename): 
        '''Loads parameters saved in HDF5 format to the Module.\n 
        Args: 
            filename (str): specify the filename as well as the path to the file without the file extension. (ex) path/to/filename 
        ''' 
        with h5.File(filename + '.hdf5', 'r') as file: 
            if not self._modules: 
                for i in file: 
                    self._params[i].data = np.array(file[i]) 
            else: 
                for i in file: 
                    for j in file[i]: 
                        self._modules[i]._params[j].data = np.array(file[i][j]) 
 
class Sequential(Module): 
    r'''A sequential container.\n 
    Modules will be added to it in the order they are passed in the constructor.  
 
    Examples:: 
        >>> # model can be defiened by adding Modules 
        >>> model = Sequential( 
        >>>     nn.Conv2d(1,20,5), 
        >>>     nn.ReLU(), 
        >>>     nn.Conv2d(20,64,5), 
        >>>     nn.ReLU() 
        >>>     ) 
        >>> # name for each layers can also be specified 
        >>> model = Sequential( 
        >>>     'conv1' = nn.Conv2d(1,20,5), 
        >>>     'relu1' = nn.ReLU(), 
        >>>     'conv2' = nn.Conv2d(20,64,5), 
        >>>     'relu2' = nn.ReLU() 
        >>>     ) 
    ''' 
    def __init__(self, *args, **kwargs): 
        super().__init__() 
        for i, module in enumerate(arg): 
            if isinstance(module, Module): 
                self._modules[str(i)] = module 
        for name, module in kwargs.items(): 
            if isinstance(module, Module): 
                self._modules[name] = module 
         
    def __call__(self, x): 
        for _, module in self._modules: 
            x = module.forward(x) 
        return x 
     
    def append(self, *arg, **kwarg): 
        if len(arg) > 1 or len(kwarg) > 1: 
            raise Exception('Too much arguments were given.') 
        if isinstance(arg, Module): 
            self._modules[str(len(self._modules))] = arg 
        for name, module in kwarg.items(): 
            if isinstance(module, Module): 
                self._modules[name] = module 
        else: 
            raise Exception('Invalid argument was given. Failed to append.')
