from qualia.data.mnist import MNIST
from qualia.nn.modules import Module
from qualia.autograd import Variable, Backward
from qualia.nn.functions import linear
from qualia.optim import *
from qualia.nn.modules import MSELoss
from tqdm import tqdm 

import matplotlib.pyplot as plt
import re
import math
import numpy as np
import os.path
path = os.path.dirname(os.path.abspath(__file__))

class Layer1st(Module):
    def __init__(self):
        super().__init__()
        M = np.zeros((1, 101))
        for i in range(50):
            M[:,i] = (i+1) * 2 * math.pi
            M[:,i+50] = (i+1) * 2 * math.pi
        M[:,100] = 0.01
        b = np.zeros((1,101))
        for i in range(50):
            b[:,i] = math.pi
            b[:,i+50] = math.pi/2

        self.weight = Variable(M)
        self.bias = Variable(b)

    def forward(self, x):
        return linear(x, self.weight, self.bias) 

class Layer2nd(Module):
    def __init__(self):
        super().__init__()
        self.weight = Variable(np.random.normal(0, math.sqrt(1/101),(101, 1))*0.1) 
        self.bias = Variable(np.zeros(1))

    def forward(self, x):
        return linear(x, self.weight, self.bias)

def activate(x):
    y1 = np.sin(x.data[:,:100])
    result = Variable(np.hstack([y1, x.data[:,100:]]))
    result.set_creator(ActivateBackward(result.shape, x))
    return result

class ActivateBackward(Backward):
    def __init__(self, output_shape, var1): 
        def f(x):
            result = x[:,:100]*np.cos(var1.data[:,:100])
            result = np.hstack([result, x[:,100:]])
            return result
        super().__init__(output_shape, var1, f) 

class TimeSeries(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Layer1st()
        self.layer2 = Layer2nd()

    def forward(self, x):
        x = activate(self.layer1(x))
        x = self.layer2(x)
        return x

model = TimeSeries()
criterion = MSELoss()

def LoadARFF(filename):
    path = os.path.dirname(os.path.abspath(__file__))
    file = open(path + '/' + filename, 'r').read()
    data = re.split(r'@DATA', file)[1]
    data = data.replace('\n',',')
    lines = data.split(',')[2:-1]
    x = np.array([data.split(',')[1]])
    for i in lines:
        x = np.append(x, i).astype(float) 
    return x

labels = LoadARFF('labor_stats.arff')

def trainer(model, criterion, optimizer, epochs, filename, load_weights=False): 
    if os.path.exists(filename+'.hdf5') and load_weights:
        model.load(filename)
        print('[*] weights loaded.')
    print('[*] training...') 
    
    for e in range(epochs):
        for i in range(256):
            data = Variable(np.array(i/256).reshape(1,1), requires_grad=False)
            label = Variable(np.array(labels[i]).reshape(1,1), requires_grad=False)
            output = model(data) 
            loss = criterion(output, label)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        prediction = []
        for i in range(356):
            data = Variable(np.array(i/256).reshape(1,1), requires_grad=False)
            output = model(data) 
            prediction.append(output.data[0,0])
        
        plt.clf()
        plt.title('prediction of time series')
        plt.xlabel('time')
        plt.axvline(x=256)
        time = np.arange(len(labels[:356]))
        data, = plt.plot(time, labels[:356], 'b', label='data')
        pred, = plt.plot(time, prediction, 'r', label='predictions')
        legend = plt.legend(handles=[data, pred], loc=0)
        plt.gca().add_artist(legend)
        plt.draw()
        plt.pause(.01)
    print('[*] training completed.')
    model.save(filename)

optim = Adadelta(model.params)
trainer(model, criterion, optim, 300, path+'/time', False)

model.load(path+'/time')
prediction = []
for i in range(356):
    data = Variable(np.array(i/256).reshape(1,1), requires_grad=False)
    output = model(data) 
    prediction.append(output.data[0,0])

plt.title('prediction of time series')
plt.xlabel('time')
plt.axvline(x=256)
time = np.arange(len(labels[:356]))
data, = plt.plot(time, labels[:356], 'b', label='data')
pred, = plt.plot(time, prediction, 'r', label='predictions')
legend = plt.legend(handles=[data, pred], loc=0)
plt.gca().add_artist(legend)
plt.show()
