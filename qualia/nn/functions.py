# -*- coding: utf-8 -*- 
import numpy as np 
from ..autograd import Variable, Backward, FunctionGenerator 
 
def exp(x): 
    '''Calculate the exponential of all elements in the input array.\n 
    Args: 
        x (Variable): input values 
    Returns: 
        (variable): Output array, element-wise exponential of x. 
    ''' 
    result = Variable(np.exp(x.data)) 
    result.set_creator(ExpBackward(result.shape, x)) 
    return result 
 
class ExpBackward(Backward): 
    def __init__(self, output_shape, var1): 
        super().__init__(output_shape, var1, (lambda x: np.exp(self.var[0].data)*x)) 
 
def ln(x): 
    '''Natural logarithm\n 
    The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x.  
    The natural logarithm is logarithm in base e. 
    Args: 
        x (Variable): input values 
    Returns: 
        (Variable): the natural logarithm of x, element-wise. 
    ''' 
    result = Variable(np.log(x.data)) 
    result.set_creator(LnBackward(result.shape, x)) 
    return result 
 
class LnBackward(Backward): 
    def __init__(self, output_shape, var1): 
        super().__init__(output_shape, var1, (lambda x: x/self.var[0].data)) 
 
def log(x): 
    '''base 10 logarithm\n 
    Return the base 10 logarithm of the input array, element-wise. 
    Args: 
        x (Variable): input values 
    Returns: 
        (Variable): The logarithm to the base 10 of x, element-wise. 
    ''' 
    result = Variable(np.log10(x.data)) 
    result.set_creator(LogBackward(result.shape, x)) 
    return result 
 
class LogBackward(Backward): 
    def __init__(self, output_shape, var1): 
        super().__init__(output_shape, var1, (lambda x: x/np.multiply(self.var[0].data,np.log(10)))) 
     
def sqrt(x): 
    '''Square root\n 
    Return the positive square-root of an array, element-wise. 
    Args: 
        x (Variable): The values whose square-roots are required. 
    Returns: 
        (Variable): An array of the same shape as x, containing the positive square-root of each element in x. 
    ''' 
    result = Variable(np.sqrt(x.data)) 
    result.set_creator(SqrtBackward(result.shape, x)) 
    return result 
 
class SqrtBackward(Backward): 
    def __init__(self, output_shape, var1): 
        super().__init__(output_shape, var1, (lambda x: x/(2*np.sqrt(self.var[0].data)))) 
 
def cbrt(x): 
    '''Cube root\n 
    Return the cube-root of an array, element-wise. 
    Args: 
        x (Variable): The values whose cube-roots are required. 
    Returns: 
        (Variable): An array of the same shape as x, containing the cube cube-root of each element in x. 
    ''' 
    result = Variable(np.cbrt(x.data)) 
    result.set_creator(CbrtBackward(result.shape, x)) 
    return result 
 
class CbrtBackward(Backward): 
    def __init__(self, output_shape, var1): 
        super().__init__(output_shape, var1, (lambda x: x/(3*np.cbrt(self.var[0].data**2)))) 
 
def sin(x): 
    '''Trigonometric sine, element-wise.\n 
    Args: 
        x (Variable): Input angle, in radians 
    Returns: 
        (Variable): The sine of each element of x. 
    ''' 
    result = Variable(np.sin(x.data)) 
    result.set_creator(SinBackward(result.shape, x)) 
    return result 
 
class SinBackward(Backward): 
    def __init__(self, output_shape, var1): 
        super().__init__(output_shape, var1, (lambda x: x*np.cos(self.var[0].data))) 
     
def cos(x): 
    '''Trigonometric cosine, element-wise.\n 
    Args: 
        x (Variable): Input angle, in radians 
    Returns: 
        (Variable): The cosine of each element of x. 
    ''' 
    result = Variable(np.cos(x.data)) 
    result.set_creator(CosBackward(result.shape, x)) 
    return result 
 
class CosBackward(Backward): 
    def __init__(self, output_shape, var1): 
        super().__init__(output_shape, var1, (lambda x: -x*np.sin(self.var[0].data))) 
 
def tan(x): 
    '''Trigonometric tangent, element-wise.\n 
    Args: 
        x (Variable): Input angle, in radians 
    Returns: 
        (Variable): The tangent of each element of x. 
    ''' 
    result = Variable(np.tan(x.data)) 
    result.set_creator(TanBackward(result.shape, x)) 
    return result 
 
class TanBackward(Backward): 
    def __init__(self, output_shape, var1): 
        super().__init__(output_shape, var1, (lambda x: x/np.cos(self.var[0].data)**2)) 
 
def arcsin(x): 
    '''Inverse sine, element-wise.\n 
    Args: 
        x (Variable): y-coordinate on the unit circle. 
    Returns: 
        (Variable): The inverse sine of each element in x, in radians and in the closed interval [-pi/2, pi/2]. 
    ''' 
    result = Variable(np.arcsin(x.data)) 
    result.set_creator(ArcsinBackward(result.shape, x)) 
    return result 
 
class ArcsinBackward(Backward): 
    def __init__(self, output_shape, var1): 
        super().__init__(output_shape, var1, (lambda x: x/np.sqrt(1-self.var[0].data**2))) 
 
def arccos(x): 
    '''Inverse cosine, element-wise.\n 
    Args: 
        x (Variable): x-coordinate on the unit circle. For real arguments, the domain is [-1, 1]. 
    Returns: 
        (Variable): The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi]. 
    ''' 
    result = Variable(np.arccos(x.data)) 
    result.set_creator(ArccosBackward(result.shape, x)) 
    return result 
 
class ArccosBackward(Backward): 
    def __init__(self, output_shape, var1): 
        super().__init__(output_shape, var1, (lambda x: -x/np.sqrt(1-self.var[0].data**2))) 
 
def arctan(x): 
    '''Inverse tangent, element-wise.\n 
    Args: 
        x (Variable): Input array 
    Returns: 
        (Variable): Out has the same shape as x. Its real part is in [-pi/2, pi/2] (arctan(+/-inf) returns +/-pi/2) 
    ''' 
    result = Variable(np.arctan(x.data)) 
    result.set_creator(ArctanBackward(result.shape, x)) 
    return result 
 
class ArctanBackward(Backward): 
    def __init__(self, output_shape, var1): 
        super().__init__(output_shape, var1, (lambda x: x/(1+self.var[0].data**2))) 
 
def sinh(x): 
    '''Compute hyperbolic sine, element-wise.\n 
    Args: 
        x (Variable): Input array 
    Returns: 
        (Variable): The corresponding hyperbolic sine values. 
    ''' 
    result = Variable(np.sinh(x.data)) 
    result.set_creator(SinhBackward(result.shape, x)) 
    return result 
 
class SinhBackward(Backward): 
    def __init__(self, output_shape, var1): 
        def f(x): 
            return x*np.cosh(self.var[0].data) 
        super().__init__(output_shape, var1, f) 
 
def cosh(x): 
    '''Compute hyperbolic cosine, element-wise.\n 
    Args: 
        x (Variable): Input array 
    Returns: 
        (Variable): The corresponding hyperbolic cosine values. 
    ''' 
    result = Variable(np.cosh(x.data)) 
    result.set_creator(CoshBackward(result.shape, x)) 
    return result 
 
class CoshBackward(Backward): 
    def __init__(self, output_shape, var1): 
        def f(x): 
            return x*np.sinh(self.var[0].data) 
        super().__init__(output_shape, var1, f) 
 
def tanh(x): 
    '''Compute hyperbolic tangent element-wise.\n 
    Args: 
        x (Variable): Input array 
    Returns: 
        (Variable): The corresponding hyperbolic tangent values. 
    ''' 
    result = Variable(np.tanh(x.data)) 
    result.set_creator(TanhBackward(result.shape, x, result.data)) 
    return result 
 
class TanhBackward(Backward): 
    def __init__(self, output_shape, var1, tmp): 
        super().__init__(output_shape, var1, (lambda x: x*(1-tmp**2))) 
 
def relu(x): 
    '''Rectified Linear Unit\n 
    Activation function that will returns max(0, x) or x+ 
    Args: 
        x (Variable): Input array 
    Returns: 
        (Variable): 0 if x<0 otherwise x   
    ''' 
    mask = (x.data <= 0) 
    tmp = x.data.copy() 
    tmp[mask] = 0 
    result = Variable(tmp) 
    result.set_creator(ReluBackward(result.shape, x)) 
    return result 
 
class ReluBackward(Backward): 
    def __init__(self, output_shape, var1): 
        def f(x): 
            mask = (var1.data <= 0) 
            x[mask] = 0 
            return x 
        super().__init__(output_shape, var1, f) 
 
def leakyrelu(x): 
    '''Leaky Rectified Linear Unit\n 
    Args: 
        x (Variable): Input array 
    Returns: 
        (Variable): 0.01x if x<0 otherwise x   
    ''' 
    mask = (x.data <= 0) 
    tmp = x.data.copy() 
    tmp[mask] = 0.01*tmp[mask] 
    result = Variable(tmp) 
    result.set_creator(LeakyReluBackward(result.shape, x)) 
    return result 
 
class LeakyReluBackward(Backward): 
    def __init__(self, output_shape, var1): 
        def f(x): 
            mask = (var1.data <= 0) 
            x[mask] = 0.01*x[mask] 
            return x 
        super().__init__(output_shape, var1, f) 
 
def elu(x, a): 
    '''Exponential Linear Unit\n 
    Args: 
        x (Variable): input array 
        a (float): a positive parameter that scales the negative region of the input so that the output region will be [-a, +inf] 
    Returns: 
        (Variable): a(e^x-1) if x<0 otherwise x   
    ''' 
    mask = (x.data <= 0) 
    tmp = x.data.copy() 
    tmp[mask] = a*(np.exp(tmp[mask])-1) 
    result = Variable(tmp) 
    result.set_creator(EluBackward(result.shape, x, tmp, a)) 
    return result 
 
class EluBackward(Backward): 
    def __init__(self, output_shape, var1, tmp, a): 
        def f(x): 
            mask = (var1.data <= 0) 
            x[mask] = x[mask]*tmp[mask] + x[mask]*a 
            return x 
        super().__init__(output_shape, var1, f) 
 
def sigmoid(x): 
    '''A logistic function\n 
    Args: 
        x (Variable): Input array 
    Returns: 
        (Variable): activated inputs   
    ''' 
    result = Variable(1/(1+np.exp(-x.data))) 
    result.set_creator(SigmoidBackward(result.shape, x, result.data)) 
    return result 
 
class SigmoidBackward(Backward): 
    def __init__(self, output_shape, var1, tmp): 
        super().__init__(output_shape, var1, (lambda x: x*tmp*(1-tmp))) 
 
def transpose(a, axes): 
    '''Permute the dimensions of an array.\n 
    Args: 
        a (Variable): Input array 
        axes (tuple of ints): permute the axes according to the values given. 
    Returns: 
        (Variable): a with its axes permuted. 
    ''' 
    result = Variable(np.transpose(a.data, axes)) 
    result.set_creator(TransposeBackward(result.shape, a, axes)) 
    return result 
 
class TransposeBackward(Backward): 
    def __init__(self, output_shape, var1, axes): 
        def f(x): 
            return np.transpose(x, [axes.index(i) for i in range(len(axes))]) 
        super().__init__(output_shape, var1, f) 
             
def tensordot(a, b, axes=1): 
    '''Compute tensor dot product along specified axes for arrays >= 1-D.\n 
    Given two tensors (arrays of dimension greater than or equal to one),  
    a and b, and an array_like object containing two array_like objects, (a_axes, b_axes),  
    sum the products of a’s and b’s elements (components) over the axes specified by a_axes and b_axes.  
    The third argument can be a single non-negative integer_like scalar, N; 
    if it is such, then the last N dimensions of a and the first N dimensions of b are summed over. 
    Args: 
        a (Variable): Tensor to dot 
        b (Variable): Tensor to dot 
        axes (int|tuple of ints): axes that will be reduced 
    Returns: 
        (Variable): dotted Tensor 
    ''' 
    result = Variable(np.tensordot(a.data, b.data, axes=axes)) 
    result.set_creator(TensordotBackward(result.shape,a,b,axes)) 
    return result 
 
class TensordotBackward(Backward): 
    def __init__(self, output_shape, var1, var2, axes): 
        need_transpose = False 
        if type(axes) is tuple: 
            need_transpose = True 
            if type(axes[0]) is int: 
                axes = ((axes[0],),(axes[1],)) 
            tp_a = [i for i in range(len(var1.shape)) if i not in axes[0]] + list(axes[0]) 
            tp_b = list(axes[1]) + [i for i in range(len(var2.shape)) if i not in axes[1]] 
            axes = len(axes[0]) 
        rev_b = [i for i in range(len(var2.shape))][axes:] 
        rev_a = [i for i in range(len(var1.shape))][:-axes] 
 
        def f1(c): 
            rev_c = [i for i in range(len(c.shape))][-len(rev_b):] 
            if not need_transpose: 
                return np.tensordot(c, self.var[1].data, axes=(rev_c,rev_b))  
            else: 
                result = np.tensordot(c, np.transpose(self.var[1].data, tp_b), axes=(rev_c,rev_b)) 
                return np.transpose(result, [tuple(tp_a).index(i) for i in range(len(tp_a))]) 
        def f2(c): 
            if not need_transpose: 
                return np.tensordot(self.var[0].data, c, axes=(rev_a,rev_a))  
            else: 
                result = np.tensordot(np.transpose(self.var[0].data,tp_a), c, axes=(rev_a,rev_a)) 
                return np.transpose(result, [tuple(tp_b).index(i) for i in range(len(tp_b))]) 
        super().__init__(output_shape, var1, var2, f1, f2) 
 
def sum(x, axis=1): 
    '''Sum of array elements over a given axis.\n 
    Args: 
        x (Variable): Elements to sum. 
        axis (int|tuple of ints): Axis or axes along which a sum is performed. Default is 1. 
    Returns: 
        (Variable): An array with the same shape as x, with the specified axis removed. 
    ''' 
    result = Variable(np.sum(x.data, axis=axis)) 
    result.set_creator(SumBackward(result.shape, x, axis)) 
    return result 
 
class SumBackward(Backward): 
    def __init__(self, output_shape, var1, axis): 
        if type(axis) is not tuple: 
            axis = (axis,)  
        shape = list(var1.shape) 
        def f(x): 
            for i in axis: 
                x = np.expand_dims(x,axis=i) 
            return np.tile(x,[shape[i] if i in axis else 1 for i in range(len(shape))])             
        super().__init__(output_shape, var1, f) 
 
def mean(x, axis=0, keepdims=False): 
    '''Average of array elements over a given axis.\n 
    Args: 
        x (Variable): Input array. 
        axis (int|tuple of ints): Axis or axes along which a average is performed. Default is 0. 
        keepdims (bool): If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. 
    Returns: 
        (Variable): An array with the same shape as x, with the specified axis removed. 
    ''' 
    result = Variable(np.mean(x.data, axis=axis, keepdims=keepdims)) 
    result.set_creator(MeanBackward(result.shape, x, axis, keepdims)) 
    return result 
 
class MeanBackward(Backward): 
    def __init__(self, output_shape, var1, axis, keepdims): 
        from functools import reduce 
        import operator 
        if type(axis) is not tuple: 
            axis = (axis,)  
        shape = list(var1.shape) 
        div = reduce(operator.mul, [shape[i] if i in axis else 1 for i in range(len(shape))], 1) 
        def f(x): 
            if not keepdims: 
                for i in axis: 
                    x = np.expand_dims(x,axis=i) 
            result = np.tile(x,[shape[i] if i in axis else 1 for i in range(len(shape))]) 
            return result/div 
        super().__init__(output_shape, var1, f) 
 
def concat(*args, axis=1):
    '''Join a sequence of arrays along an existing axis.\n 
    Args: 
        *args (Variable): The arrays must have the same shape, except in the dimension corresponding to axis.
        axis (int): The axis along which the arrays will be joined. Default is 1. 
    Returns: 
        (Variable): The concatenated array. 
    ''' 
    result = Variable(np.concatenate(tuple(i.data for i in args), axis)) 
    result.set_creator(ConcatBackward(result.shape, axis, *args)) 
    return result 

class ConcatBackward(Backward):
    def __init__(self, output_shape, axis, *args):
        s = [i.shape[axis] for i in args]
        split = [np.sum(s[n] for n in range(i+1)) for i in range(len(s))]
        class Gen(FunctionGenerator):
            def __iter__(self):
                for i in range(len(split)):
                    yield lambda x: np.split(x,split,axis=axis)[i]
        super().__init__(output_shape, *args, Gen()) 

def repeat(x, num):
    '''Repeats input 
    Args:
        x (Variable): Input array
        num (int): The number of repetitions for each element.
    Returns:
        (Variable_1,...,Variable_num): identical variables
    '''
    tmp = np.repeat(x.data.reshape(1,*x.shape), num, axis=0)
    result = [Variable(tmp[i]) for i in range(num)]
    backward = RepeatBackward(x.shape, x, num)
    for i in result:
        i.set_creator(backward) 
    return result 
 
class RepeatBackward(Backward):
    def __init__(self, output_shape, var1, num):
        self.creator = var1.creator
        self.counter = 0
        def f(x):
            if self.counter == 0:
                var1.creator = None
                self.counter += 1
                return x
            elif self.counter == num - 1:
                var1.creator = self.creator
                return var1.grad + x
            else:
                self.counter += 1
                return var1.grad + x
        super().__init__(output_shape, var1, f)        
        
def reshape(x, shape): 
    '''Reshape a Variable\n 
    Args: 
        x (Variable): Input array 
        shape (tuple of int): desired output shape 
    Returns: 
        (Variable): rehaped array 
    ''' 
    result = Variable(x.data.reshape(shape)) 
    result.set_creator(ReshapeBackward(result.shape, x)) 
    return result 
 
class ReshapeBackward(Backward): 
    def __init__(self, output_shape, var1): 
        def f(arg): 
            if arg.shape != output_shape: 
                axis=[] 
                for i in range(len(arg.shape)): 
                    if arg.shape[i] != output_shape[i] and output_shape[i] != -1:  
                        axis.append(i) 
                arg = np.sum(arg, axis=tuple(axis)) 
            return arg.reshape(var1.shape) 
        super().__init__(output_shape, var1, f) 
 
def linear(x, weight, bias=None): 
    tmp = tensordot(x, weight, axes=1) 
    if bias is not None: 
        shape = np.ones_like(tmp.shape) 
        shape[-1] = tmp.shape[-1] 
        result = tmp + reshape(bias, tuple(shape)) 
    else: 
        result = tmp 
    return result 
 
# TODO 
def bilinear(x1, x2, weight, bias=None): 
    raise NotImplementedError 
 
def conv2d(x, kernel, bias=None, stride=1, padding=0, dilation=1): 
    '''Applies a 2D convolution over an input signal composed of several input planes.\n 
    Args: 
        x (Variable): Input tensor with shepe of [batch, channel, height, width] 
        kernel (Variable): Kernel with shape of [patch, channel, kernel_height, kernel_width] 
        bias (Variable): Bias with shape of [patch] to add if needed. Default: None 
        stride (int|tuple of int): Stride of the convolution. Default: 1 
        padding (int|tuple of int): Padding controls the amount of implicit zero-paddings on both sides for padding number of points for each dimension. Default: 0 
        dilation (int|tuple of int): Spacing between kernel elements. Default: 1 
    Returns: 
        (Variable): Output tensor will have shape of [batch, patch, out_height, out_width] 
    ''' 
    if type(stride) is int: 
        stride = (stride, stride) 
    if type(padding) is tuple: 
        if len(padding) > 2: 
            raise Exception('too much arguments were givin for padding') 
        padding = ((0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])) 
    if type(padding) is int: 
        padding = ((0,0),(0,0),(padding, padding),(padding, padding)) 
    if type(dilation) is int: 
        dilation = (dilation, dilation) 
     
    batch, _, height, width = x.shape 
    patch, _, kernel_height, kernel_width = kernel.shape 
 
    oh = int((height+2*padding[2][0]-dilation[0]*(kernel_height-1)-1)/stride[0]+1) 
    ow = int((width+2*padding[3][0]-dilation[1]*(kernel_width-1)-1)/stride[1]+1) 
 
    padded = np.pad(x.data, padding, 'constant') 
    reshaped = _reshape_img(padded, batch, oh, ow, kernel.shape, stride, dilation) 
 
    tmp = Variable(np.tensordot(reshaped, kernel.data, ((2,3,4),(1,2,3))).transpose(0,2,1).reshape(-1,patch,oh,ow)) 
    tmp.set_creator(Conv2dBackward(tmp.shape, x, kernel, reshaped, stride, padded.shape, dilation, oh, ow)) 
    if bias is not None: 
        result = tmp + reshape(bias, (1,-1,1,1)) 
    else: 
        result = tmp 
    return result 
 
def _reshape_img(x, batch, oh, ow, kernel_shape, stride, dilation): 
    '''Reshape the input image to simplify the convolutional operation 
    Args: 
        x (numpy.ndarray): input image 
        batch (int): size of batch 
        oh (int): size of output height 
        ow (int): size of output width 
        kernel_shape (tuple of int): shape of the kernel [patch, channel, height, width] 
        stride (tuple of int): stride of convolution in 2D 
        dilation (tuple of int): spacing between kernel in 2D 
    Returns: 
        (numpy.ndarray): reshaped image with shape of [batch, oh*ow, channel, height, width] 
    ''' 
    _, channel, height, width = kernel_shape 
    fh, fw = ((height-1)*dilation[0]+1, (width-1)*dilation[1]+1) 
    result = np.zeros((batch, oh*ow, channel, height, width)) 
    for i in range(oh): 
        for j in range(ow): 
            tmp = x[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw] 
            result[:, i*ow+j, :, :, :] = tmp[:, :, ::dilation[0], ::dilation[1]] 
    return result 
 
class Conv2dBackward(Backward): 
    def __init__(self, output_shape, var1, var2, reshaped, stride, padded_shape, dilation, oh, ow): 
        def f1(arg): 
            batch, patch, _, _ = arg.shape 
            delta = np.tensordot(arg.reshape(batch,patch,-1), self.var[1].data, (1,0)) 
            return self._rev_img(delta, oh, ow, var1.shape, var2.shape, stride, padded_shape, dilation) 
        def f2(arg): 
            batch, patch, _, _ = arg.shape 
            return np.tensordot(arg.reshape(batch,patch,-1), reshaped, ((0,2),(0,1))) 
        super().__init__(output_shape, var1, var2, f1, f2) 
     
    def _rev_img(self, delta, oh, ow, x_shape, kernel_shape, stride, padded_shape, dilation): 
        batch, channel, h, w = x_shape 
        _, _, height, width = kernel_shape 
        _, _, ph, pw = padded_shape 
        fh, fw = ((height-1)*dilation[0]+1, (width-1)*dilation[1]+1) 
        result = np.zeros(padded_shape) 
        for i in range(oh): 
            for j in range(ow): 
                tmp = np.zeros((batch, channel, fh, fw)) 
                tmp[:, :, ::dilation[0], ::dilation[1]] = delta[:, i*ow+j, :, :, :] 
                result[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw] += tmp 
        if ph == h and pw == w: 
            return result 
        elif ph == h and pw != w: 
            return result[:,:,:,int((pw-w)/2):-int((pw-w)/2)] 
        elif ph != h and pw == w: 
            return result[:,:,int((ph-h)/2):-int((ph-h)/2),:] 
        else: 
            return result[:,:,int((ph-h)/2):-int((ph-h)/2),int((pw-w)/2):-int((pw-w)/2)] 
 
# TODO 
def conv_transpose2d(x, kernel, bias=None, stride=1, padding=0, output_padding=0, dilation=1): 
    '''Applies a 2D transposed convolution operator over an input image composed of several input planes.\n 
    Args: 
        x (Variable): Input tensor with shepe of [batch, channel, height, width] 
        kernel (Variable): Kernel with shape of [patch, channel, kernel_height, kernel_width] 
        bias (Variable): Bias with shape of [patch] to add if needed. Default: None 
        stride (int|tuple of int): Stride of the convolution. Default: 1 
        padding (int|tuple of int): zero-padding will be added to both sides of each dimension in the input. Default: 0 
        output_padding (int|tuple of int): Additional size added to one side of each dimension in the output shape. Default: 0 
        dilation (int|tuple of int): Spacing between kernel elements. Default: 1 
    Returns: 
        (Variable): Output tensor will have shape of [batch, patch, out_height, out_width] 
    ''' 
    if type(stride) is int: 
        stride = (stride, stride) 
    if type(padding) is tuple: 
        if len(padding) > 2: 
            raise Exception('too much arguments were givin for padding') 
        padding = ((0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])) 
    if type(padding) is int: 
        padding = ((0,0),(0,0),(padding, padding),(padding, padding)) 
    if type(output_padding) is tuple: 
        if len(output_padding) > 2: 
            raise Exception('too much arguments were givin for output_padding') 
        output_padding = ((0,0),(0,0),(output_padding[0],0),(output_padding[1],0)) 
    if type(output_padding) is int: 
        output_padding = ((0,0),(0,0),(output_padding, 0),(output_padding, 0)) 
    if type(dilation) is int: 
        dilation = (dilation, dilation) 
     
    batch, _, height, width = x.shape 
    patch, _, kernel_height, kernel_width = kernel.shape 
 
    oh = int((height-1)*stride[0]-2*padding[2][0]+kernel_height+output_padding[2][0]) 
    ow = int((width-1)*stride[1]-2*padding[3][0]+kernel_width+output_padding[3][0]) 
 
    padded = np.pad(x.data, padding, 'constant') 
     
 
# TODO 
class ConvTranspose2dBackward(Backward): 
    def __init__(self): 
        pass 
 
def max_pool2d(x, kernel_size, stride=None, padding=0, dilation=1): 
    '''Applies a 2D max pooling over an input signal composed of several input planes.\n 
    Args: 
        x (Variable): [N,C,H,W] 
        kernel_size (tuple of int): the size of the window to take a max over 
        stride (int|tuple of int): the stride of the window. Default: kernel_size 
        padding (int|tuple of int): implicit zero padding to be added on both sides 
        dilation (int|tuple of int): a parameter that controls the stride of elements in the window 
    Returns: 
        (Variable): [N,C,H_out,W_out] 
    ''' 
    if type(stride) is int: 
        stride = (stride, stride) 
    if stride is None: 
        stride = kernel_size 
    if type(padding) is tuple: 
        if len(padding) > 2: 
            raise Exception('too much arguments were givin for padding') 
        padding = ((0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])) 
    if type(padding) is int: 
        padding = ((0,0),(0,0),(padding, padding),(padding, padding)) 
    if type(dilation) is int: 
        dilation = (dilation, dilation) 
 
    batch, channel, height, width = x.shape 
    kernel_height, kernel_width = kernel_size 
 
    oh = int((height+2*padding[2][0]-dilation[0]*(kernel_height-1)-1)/stride[0]+1) 
    ow = int((width+2*padding[3][0]-dilation[1]*(kernel_width-1)-1)/stride[1]+1) 
 
    padded = np.pad(x.data, padding, 'constant') 
    reshaped = _reshape_img(padded, batch, oh, ow, (1,channel)+kernel_size, stride, dilation) 
 
    tmp, save_for_back = map(lambda f: f(reshaped.reshape(batch, oh*ow, channel, -1), axis = 3).reshape(batch, channel, oh, ow), [np.max, np.argmax]) 
    result = Variable(tmp) 
    result.set_creator(Maxpool2dBackward(result.shape, x, kernel_size, save_for_back, stride, padded.shape, dilation, oh, ow)) 
    return result 
 
class Maxpool2dBackward(Backward): 
    def __init__(self, output_shape, var1, kernel_size, argmax, stride, padded_shape, dilation, oh, ow): 
        def f(arg): 
            return self._rev_img(arg, kernel_size,argmax, oh, ow, var1.shape, stride, padded_shape, dilation) 
        super().__init__(output_shape, var1, f) 
     
    def _rev_img(self, delta, kernel_size, argmax, oh, ow, x_shape, stride, padded_shape, dilation): 
        batch, channel, h, w = x_shape 
        kernel_height, kernel_width = kernel_size 
        _, _, ph, pw = padded_shape 
        fh, fw = ((kernel_height-1)*dilation[0]+1, (kernel_width-1)*dilation[1]+1) 
        result = np.zeros(padded_shape) 
        for i in range(oh): 
            for j in range(ow): 
                tmp = np.zeros((batch, channel, fh, fw)) 
                tmp[:, :, ::dilation[0], ::dilation[1]].reshape(batch, channel,-1)[:,:,argmax[:,:,i,j]] = delta[:,:,i,j] 
                result[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw] += tmp 
        if ph == h and pw == w: 
            return result 
        elif ph == h and pw != w: 
            return result[:,:,:,int((pw-w)/2):-int((pw-w)/2)] 
        elif ph != h and pw == w: 
            return result[:,:,int((ph-h)/2):-int((ph-h)/2),:] 
        else: 
            return result[:,:,int((ph-h)/2):-int((ph-h)/2),int((pw-w)/2):-int((pw-w)/2)] 
 
def normalize(x, axis=1, eps=1e-10): 
    '''Normalize the incoming data\n 
    Args: 
        x (Variable): input array 
        axis (int|tuple of int): axis to compute the mean and variance  
        eps (float): a small value to stablize the computation 
    Returns: 
        (Variable): normalized array 
    ''' 
    ave = mean(x, axis=axis, keepdims=True) 
    var = mean((x-ave)**2, axis=axis, keepdims=True) 
    result = (x-ave)/sqrt(var+eps) 
    return result 
