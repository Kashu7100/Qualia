<p align="center">
  <img src="https://kashu.ml/wp-content/uploads/2018/08/qualia-1-700x379.png?raw=true" alt="Qualia Logo"/>
</p>

Qualia is a deep learning framework deeply integrated with autograd designed for maximum flexibility. Qualia features in automatic differentiation and dynamic graphing.


Physicalism, which considers there is nothing over or above the physical, has been criticized by Thomas Nagel (What is it like to be a bat?), Frank C. Jackson (Mary's Room),  David J. Chalmers (Philosophical zombie) and others mainly because it lacks descriptions for the hard problem of consciousness. In the paper "[Absent Qualia, Fading Qualia, Dancing Qualia](http://consc.net/papers/qualia.html)," David argues that if a system reproduces the functional organization of the brain, it will also reproduce the qualia associated with the brain. This library "Qualia" named after the series of arguments in philosophy of mind associated with the qualia, hoping the creation of a system with subjective consciousness. 
 
 

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

'''
$ python setup.py install
'''

## Examples

The following example will compute the Sum of Squared Error 
```bash
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
