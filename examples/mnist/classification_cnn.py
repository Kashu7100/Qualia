from qualia.data.mnist import MNIST
from qualia.nn.modules import Module, Conv2d, MaxPool2d, Linear
from qualia.nn.functions import leakyrelu, reshape
from qualia.optim import *
from qualia.nn.modules import CrossEntropyLoss
from qualia.util import trainer, tester

import os.path
path = os.path.dirname(os.path.abspath(__file__))

class DeepConvNet(Module):
    def __init__(self):
        super().__init__()
        # (N,1,28,28) -> (N,16,28,28)
        self.conv1 = Conv2d(1, 16, 5, padding=2)
        # (N,16,28,28) -> (N,16,14,14)
        self.pool1 = MaxPool2d((2,2))
        # (N,16,14,14) -> (N,16,14,14)
        self.conv2 = Conv2d(16, 16, 5, padding=2)
        # (N,16,14,14) -> (N,16,7,7)
        self.pool2 = MaxPool2d((2,2))
        # Reshape
        self.linear1 = Linear(16*7*7, 128)
        self.linear2 = Linear(128, 10)
    
    def forward(self, x):
        x = leakyrelu(self.conv1(x))
        x = self.pool1(x)
        x = leakyrelu(self.conv2(x))
        x = self.pool2(x)
        x = reshape(x, (-1, 16*7*7))
        x = leakyrelu(self.linear1(x))
        x = leakyrelu(self.linear2(x))
        return x

mnist = MNIST(flatten=False)
model = DeepConvNet()
criterion = CrossEntropyLoss(live_plot=False)
optim = Momentum(model.params, lr=0.001)
trainer(model, criterion, optim, mnist, 30, 1, path+'/mnist_cnn', False)
tester(model, criterion, mnist, 16, path+'/mnist_cnn', True)
