<p align="center">
  <img src="https://kashu.ml/wp-content/uploads/2018/08/qualia-1-700x379.png?raw=true" alt="Qualia Logo"/>
</p>

Qualia is a deep learning framework deeply integrated with autograd designed for maximum flexibility. Qualia features in automatic differentiation and dynamic graphing.

### introduction

Physicalism, which considers there is nothing over or above the physical, has been criticized by Thomas Nagel (What is it like to be a bat?), Frank C. Jackson (Mary's Room),  David J. Chalmers (Philosophical zombie) and others mainly because it lacks descriptions for the hard problem of consciousness. In the paper "[Absent Qualia, Fading Qualia, Dancing Qualia](http://consc.net/papers/qualia.html)," David argues that if a system reproduces the functional organization of the brain, it will also reproduce the qualia associated with the brain. This library "Qualia" named after the series of arguments in philosophy of mind associated with the qualia, hoping for the creation of a system with subjective consciousness. 

### overview

Qualia is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **qualia** | a deep learning framework utilizing NumPy |
| **qualia.autograd** | a wrapper class for NumPy that supports a dynamic automatic differentiation|
| **qualia.nn** | a neural networks library deeply integrated with autograd designed for maximum flexibility |
| **qualia.data** | provides datasets for handy testing |
| **qualia.optim** | optimizers for training neural nets |
| **qualia.util** | DataLoader, Trainer and other utility functions for convenience |

## Requirements

* Python 3.x
* numpy
* matplotlib 
* h5py 
* tqdm

## Installation
```
$ python setup.py install
```

## Examples

### classification with CNN

Classification of handwritten digits done on [mnist](http://yann.lecun.com/exdb/mnist/) dataset using CNN

![source code](https://github.com/Kashu7100/Qualia/blob/master/examples/mnist/classification_cnn.py)

<p align="center">
  <img src="assets/mnist.PNG" height="400"/>
</p>

### PCA with non-linear autoencoder

Dimensionality Reduction is a technique widely used in data science to visualize data.
The following image is the plot of two principal components of handwritten digits found by non-linear autoencoder.

<p align="center">
  <img src="assets/mnist_ae.PNG" height="450"/>
</p>

### time series prediction

Regression of a neural network with sinusoidal activation functions done on labor_stats data

![source code](examples/timeseries/timeseries.py)

<p align="center">
  <img src="assets/labor_stats_pred.png" height="450"/>
</p>

## Tutorials 

### automatic differentiation (AD)
Fundamental to [AD](https://en.wikipedia.org/wiki/Automatic_differentiation) is the decomposition of differentials provided by the chain rule. Qualia implements the reverse accumulation AD in qualia.autograd.

In the example code of this tutorial, we assume for simplicity that the following symbols are already imported.
```python
import numpy as np
from qualia.autograd import Variable
```
Qualia uses the so called “Define-by-Run” scheme, so forward computation itself defines the network. By using a Variable object, Qualia can keep track of every operation. Here, the resulting y is also a Variable object.
```python
x = Variable(np.array([5]))
y = x**2 - 2*x + 1
# prints result of the computation: y = 16
print(y.data)
```
At this moment we can compute the derivative.
```python
y.backward()
# prints gradient of x: x = 8 
print(x.grad)
```
All these computations can be generalized to a multidimensional tensor input. When the output is not a scalar quantity, a tenspr  with the same dimentions as the output that is filled with ones will be given by default to start backward computation.
```python
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = x**2 - 2*x + 1
y.backward()
# prints gradients of x:
# [[ 0  2  4]
#  [ 6  8 10]]
print(x.grad)
```

### basic usage

The following example will compute the Sum of Squared Error 
```python
import numpy as np
from qualia.autograd import Variable
from qualia.nn.functions import sum
# Create Variable objects 
prediction = Variable(np.random.rand(10,3)) 
label = Variable(np.random.rand(10,3),requires_grad=False) 
# Write an equation 
loss = sum((prediction-label)**2,axis=1)/2 
# Print loss 
print('loss is: \n{}'.format(loss.data)) 
# Calclate gradiant 
loss.backward() 
# Print gradient 
print('gradiant for prediction is: \n{}'.format(prediction.grad)) 
# When requires_grad is False, gradients will be zero 
print('gradient for label is: \n{}'.format(label.grad)) 
```

## License

Source codes in the repository follows [MIT](http://www.opensource.org/licenses/MIT) license.
