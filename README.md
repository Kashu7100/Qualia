<p align="center">
  <img src="https://kashu.ml/wp-content/uploads/2018/08/qualia-1-700x379.png?raw=true" alt="Qualia Logo"/>
</p>

Qualia is a deep learning framework for a flexible modeling. Qualia is purely written in Python3 and requires few external libraries. Qualia features in automatic differentiation and dynamic graphing.

| Component | Description |
| ---- | --- |
| **qualia** | a deep learning framework utilizing NumPy |
| **qualia.autograd** | a wrapper class for NumPy that supports a dynamic automatic differentiation|
| **qualia.nn** | a neural networks library deeply integrated with autograd designed for maximum flexibility |
| **qualia.data** | provides datasets for handy testing |
| **qualia.optim** | optimizers for training neural nets |
| **qualia.utils** | DataLoader, Trainer and other utility functions for convenience |

## Requirements

* Python 3.x
* Numpy
* Matplotlib 

## Installation

### Windows

### Mac/Linux

## Examples

```
The following example will compute the Sum of Squared Error 
  >>> import numpy as np
  >>> from qualia.autograd import Variable
  >>> from qualia.nn.functions import sum
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
```

## License

Source codes in the repository follows [MIT](http://www.opensource.org/licenses/MIT) license.
