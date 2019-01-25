# -*- coding: utf-8 -*- 
import numpy as np 
import types 
 
class Variable(object): 
    r'''Variable 
    Wrapper class for numpy.ndarray to execute automatic differentiation 
 
    Args: 
        data (numpy.ndarray|int|float): tensor to compute the automatic differentiation 
        requires_grad (bool): Whether to store grads. If False is set, grad of the Variable will be zeros. 
     
    Attributes: 
        data (numpy.ndarray): Stores data of the Variable 
        grad (numpy.ndarray): Stores gradients of the Variable  
        creator (Backward obj): Stores the creator of the Variable, which will be called at the backpropagation. 
        requires_grad (bool): Whether to store grads. If False is set, grad of the Variable will be zeros. 
        shape (tuple): Stores the shape of Variable's data 
        ndim (int): Stores the number of Variable's data dimentions  
     
    Examples:: 
        The following example will compute the Sum of Squared Error 
        >>> # Create Variable objects 
        >>> prediction = Variable(np.random.rand(10,3)) 
        >>> label = Variable(np.random.rand(10,3),requires_grad=False) 
        >>> # Write an equation 
        >>> loss = sum((prediction-label)**2,axis=1)/2 
        >>> # Print loss 
        >>> print('loss is: \n{}'.format(loss.data)) 
        >>> # Calclate gradiant 
        >>> loss.backward() 
        >>> # Print gradient 
        >>> print('gradiant for prediction is: \n{}'.format(prediction.grad)) 
        >>> # When requires_grad is False, gradients will be zero 
        >>> print('gradient for label is: \n{}'.format(label.grad)) 
    ''' 
    def __init__(self, data, requires_grad=True): 
        if type(data) is not np.ndarray: 
            self.data = np.array([data]) 
        else: 
            self.data = data 
        self.grad = None 
        self.creator = None 
        self.requires_grad = requires_grad
 
    def set_creator(self, obj): 
        self.creator = obj 
     
    def backward(self, *args): 
        if not bool(args):
            args = [np.ones_like(self.data)]     
        self.creator.backward(*args) 
     
    def handle_const(self, const): 
        if type(const) is Variable: 
            return const 
        else: 
            return Variable(const, requires_grad=False) 

    def __setattr__(self, key, value):   
        object.__setattr__(self, key, value)
        if key == 'data':
            object.__setattr__(self, 'shape', self.data.shape)
            object.__setattr__(self, 'ndim', self.data.ndim) 
    
    def __getitem__(self, slice):
        result = Variable(self.data[slice])
        result.set_creator(SliceBackward(result.shape, self, slice))
        return result
     
    def __len__(self): 
        return self.ndim
 
    def __add__(self, other): 
        other = self.handle_const(other) 
        result = Variable(np.add(self.data, other.data)) 
        result.set_creator(AddBackward(result.shape, self, other)) 
        return result 
     
    def __radd__(self, other): 
        other = self.handle_const(other) 
        result = Variable(np.add(self.data, other.data)) 
        result.set_creator(AddBackward(result.shape, self, other)) 
        return result 
 
    def __sub__(self, other): 
        other = self.handle_const(other) 
        result = Variable(np.subtract(self.data, other.data)) 
        result.set_creator(SubBackward(result.shape, self, other)) 
        return result 
     
    def __rsub__(self, other): 
        other = self.handle_const(other) 
        result = Variable(np.subtract(other.data, self.data)) 
        result.set_creator(SubBackward(result.shape, other, self)) 
        return result 
 
    def __mul__(self, other): 
        other = self.handle_const(other) 
        result = Variable(np.multiply(self.data, other.data)) 
        result.set_creator(MulBackward(result.shape, self, other)) 
        return result 
 
    def __rmul__(self, other): 
        other = self.handle_const(other) 
        result = Variable(np.multiply(self.data, other.data)) 
        result.set_creator(MulBackward(result.shape, self, other)) 
        return result 
     
    def __matmul__(self, other): 
        result = Variable(self.data @ other.data) 
        result.set_creator(MatmulBackward(result.shape, self, other)) 
        return result 
     
    def __neg__(self): 
        result = Variable(np.multiply(-1, self.data)) 
        result.set_creator(NegBackward(result.shape, self)) 
        return result 
 
    def __truediv__(self, other): 
        other = self.handle_const(other) 
        result = Variable(np.divide(self.data,other.data)) 
        result.set_creator(DivBackward(result.shape, self, other)) 
        return result 
 
    def __rtruediv__(self, other): 
        other = self.handle_const(other) 
        result = Variable(np.divide(other.data, self.data)) 
        result.set_creator(DivBackward(result.shape, other, self)) 
        return result 
 
    def __pow__(self, other): 
        other = self.handle_const(other) 
        result = Variable(np.power(self.data, other.data)) 
        result.set_creator(PowBackward(result.shape, self, other)) 
        return result 
     
    def __rpow__(self, other): 
        other = self.handle_const(other) 
        result = Variable(np.power(other.data, self.data)) 
        result.set_creator(PowBackward(result.shape, other, self)) 
        return result 
 
class FunctionGenerator(object):
    def __iter__(self):
        raise NotImplementedError 
 
class Backward(object): 
    '''Base class for all backward classes.\n 
    All backward class should inherit this class. 
    Attributes:
        shape (tuple of int): output shape of a function
        var (tuple of Variable): Variable(s) that was feeded 
        func (tuple of Function): function(s) should deal the backward propagation
    ''' 
    def __init__(self, output_shape, *args): 
        self.shape = output_shape 
        self.var = tuple(i for i in args if type(i) is Variable) 
        self.func = tuple(i for i in args if type(i) is types.FunctionType) 
        if not bool(self.func):
            self.func = tuple(i for i in args if isinstance(i, FunctionGenerator))[0]
        if len(self.var) != len([*self.func]): 
            raise Exception('number of variables and functions should match up. Got {} var: {} and {} func: {}'.format(len(self.var), self.var, len([*self.func]), [*self.func])) 
 
    def backward(self, *args): 
        args = [self.handle_broadcast(i) for i in args]
        for func, var in zip(self.func, self.var):
            var.grad = func(*args)
            if var.creator is not None:
                var.backward(var.grad)
            if not var.requires_grad:
                var.grad = None
        
    def handle_broadcast(self, arg): 
        if arg.shape != self.shape: 
            tmp = list(self.shape) 
            if len(arg.shape) > len(self.shape): 
                for i in range(len(arg.shape)): 
                    if not arg.shape[i] in self.shape: 
                        tmp.insert(i, 1) 
            axis=[] 
            for i in range(len(arg.shape)): 
                if arg.shape[i] != tmp[i]: 
                    axis.append(i) 
            arg = np.sum(arg, axis=tuple(axis)) 
            return arg.reshape(self.shape) 
        else: 
            return arg 
         
class SliceBackward(Backward):
    def __init__(self, output_shape, var1, slice):
        def f(x):
            result = np.zeros_like(var1.data)
            result[slice] = x
            return result
        super().__init__(output_shape, var1, f)     
     
class AddBackward(Backward): 
    def __init__(self, output_shape, var1, var2): 
        super().__init__(output_shape, var1, var2, (lambda x: x), (lambda x: x)) 
 
class SubBackward(Backward): 
    def __init__(self, output_shape, var1, var2): 
        super().__init__(output_shape, var1, var2, (lambda x: x), (lambda x: -x)) 
     
class MulBackward(Backward): 
    def __init__(self, output_shape, var1, var2): 
        def f1(arg): 
            return np.multiply(arg, self.var[1].data) 
        def f2(arg): 
            return np.multiply(arg, self.var[0].data) 
        super().__init__(output_shape, var1, var2, f1, f2) 
 
class MatmulBackward(Backward): 
    def __init__(self, output_shape, var1, var2): 
        def f1(arg): 
            return arg @ self.var[1].data.T 
        def f2(arg): 
            return self.var[0].data.T @ arg 
        super().__init__(output_shape, var1, var2, f1, f2) 
 
class NegBackward(Backward): 
    def __init__(self, output_shape, var1): 
        super().__init__(output_shape, var1, (lambda x: -x)) 
 
class DivBackward(Backward): 
    def __init__(self, output_shape, var1, var2): 
        def f1(arg): 
            return arg/self.var[1].data 
        def f2(arg): 
            return -arg*(self.var[0].data/self.var[1].data**2) 
        super().__init__(output_shape, var1, var2, f1, f2) 
 
class PowBackward(Backward): 
    def __init__(self, output_shape, var1, var2): 
        def f1(arg): 
            return self.var[1].data*(self.var[0].data**(self.var[1].data - np.array([1])))*arg 
        def f2(arg): 
            mask = (self.var[0].data <= 0) 
            tmp = self.var[0].data.copy() 
            tmp[mask] = 1e-10 
            return (self.var[0].data**self.var[1].data)*np.log(tmp)*arg 
        super().__init__(output_shape, var1, var2, f1, f2) 
