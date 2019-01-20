# -*- coding: utf-8 -*- 
import os.path
import numpy as np 
import collections
import matplotlib.pyplot as plt
from matplotlib import style
from tqdm import tqdm 
from .autograd import Variable 
 
def timer(func): 
    ''' 
    decorator function that will print the excecution time of a function. 
    ''' 
    import time 
    def wrapper(*args, **kwargs): 
        start = time.time() 
        result = func(*args, **kwargs) 
        end = time.time() 
        print('[*] The function \'{}\' took {} sec to excecute.'.format(func.__name__,start-end)) 
        return result 
    return wrapper 
 
def trainer(model, criterion, optimizer, dataloader, epochs, minibatch, filename, load_weights=False): 
    ''' 
    This trainer will ease the process of training a model. 
 
    Args: 
        model (Module): model to train 
        criterion (Regression|Classification): loss function to use 
        optimizer (Optimizer): optimizer to use 
        dataloader (DataLoader): dataloader to use 
        epochs (int): number of epochs 
        minibatch (int): number of batch to use for training 
        filename (string): specify the filename as well as the saving path without the file extension. (ex) path/to/filename 
        load_weights (bool): load pre-trained weights if available. Default:False
    ''' 
    best_acc = 0
    i = 0
    if os.path.exists(filename+'.hdf5') and load_weights:
        model.load(filename)
        print('[*] weights loaded.') 
    if criterion.live_plot: 
        fig = plt.gcf()
        fig.show()
        fig.canvas.draw()
    print('[*] training...') 
    dataloader.set_batch(minibatch)
    for epoch in tqdm(range(epochs)): 
        if minibatch > 1:
            dataloader.train_dataset.shuffle()
        for data, label in tqdm(dataloader.train_dataset, ncols=150, desc='epoch {}, acc {}'.format(epoch+1, best_acc)): 
            output = model(data) 
            loss = criterion(output, label)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            if criterion.live_plot and is_fibonacci(i):
                plt.plot(fibonacci(len(criterion.losses)) ,[criterion.losses[j] for j in fibonacci(len(criterion.losses))])
                plt.pause(0.001)
                fig.canvas.draw()
            i += 1
        tmp_acc = 0
        num = 0
        for data, label in dataloader.test_dataset: 
            output = model(data) 
            tmp_acc += criterion.get_acc(output, label) 
            num += 1
        tmp_acc /= num
        if tmp_acc > best_acc:
            best_acc = tmp_acc
            model.save(filename) 
    print('[*] training completed.') 
    print('best accuracy: {}'.format(best_acc)) 

def tester(model, criterion, dataloader, minibatch, filename, show_img=False):
    '''
    This tester will ease the evaluation process of a model

    Args: 
        model (Module): model to train 
        criterion (Regression|Classification): loss function to use 
        dataloader (DataLoader): dataloader to use
        minibatch (int): number of batch to use for training 
        filename (string): specify the filename as well as the saving path without the file extension. (ex) path/to/filename 
    '''
    model.load(filename)
    acc = 0
    num = 0
    print('[*] testing...') 
    dataloader.set_batch(minibatch)
    for data, label in dataloader.test_dataset:
        output = model(data)
        if show_img:
            imgs_show(data, label, output)
        acc += criterion.get_acc(output, label)
        num += 1
    print('accuracy: {}'.format(acc/num))

def imgs_show(data, label, pred, size=(28,28), col=6):
    '''
    Draws images along it's labels and the prediction of the model
    
    Args:
        data (Variable): images to visualize
        label (Variable): the labels for the data
        pred (Variable): predicted value by the network
        size (tuple of int): image size
        col (int): number of images to display in a row
    '''
    data = data.data
    if label.shape[1] != 1:
        label = np.argmax(label.data, axis=1).reshape(-1,1) 
    else: 
        label = label.data
    if pred.shape[1] != 1:
        pred = np.argmax(pred.data, axis=1).reshape(-1,1) 
    else: 
        pred = pred.data 
    
    for i in range(data.shape[0]):
        plt.subplot(int(data.shape[0])/col+1,col,i+1)
        plt.imshow(data[i].reshape(size), cmap='gray', interpolation='nearest')
        plt.title('Ans:{}, Pred:{}'.format(label[i], pred[i]), fontsize=7)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

class DataSet(object):
    '''Dataset\n
    
    Attributes: 
        batch (int): batch size for sampling
        attributes (str): attributes of the dataset, default is None
        data (ndarray): features of the dataset, default is None
        label (ndarray): labels of the dataset, default is None
        corr (dict): dictionary that holds correration between class index and class labels
    '''
    def __init__(self):
        self.batch = 1
        self.attributes = None
        self.data = None
        self.label = None
        self.corr = None
        self.idx = 0
        
    def __add__(self, a):
        if self.data is None or a.data is None:
            raise ValueError
        if not isinstance(a, DataSet):
            raise TypeError
        dataset = DataSet()
        if self.attributes == a.attributes:
            dataset.attributes = self.attributes
        else:
            dataset.attributes = self.attributes + '\n' + a.attributes
        dataset.data = np.concatenate((self.data, a.data), axis=0)
        if self.label is not None and a.label is not None:
            if self.corr is not None:
                inv_corr1 = {v:k for k, v in self.corr.items()}
                label1 = np.array(list(map(lambda x:inv_corr1[x],self.label)))
            else:
                label1 = self.label
            if a.corr is not None:
                inc_corr2 = {v:k for k, v in a.corr.items()}
                label2 = np.array(list(map(lambda x:inv_corr2[x],a.label)))
            else:
                label2 = a.label
            dataset.label = np.concatenate((label1, label2), axis=0)
        return dataset

    def __setattr__(self, key, value): 
        if key == 'label' and value is not None:
            if np.issubdtype(value.dtype, np.string_):
                labels = collections.Counter(value)
                corr = {_key:i for i, _key in enumerate(sorted(labels.keys()))}
                object.__setattr__(self, 'corr', corr)
                tmp = list(map(lambda x:self.corr[x],value))
                tmp_label = np.zeros(len(tmp),dtype=np.int32)
                for i in range(len(tmp_label)):
                    tmp_label[i] = tmp[i]
                object.__setattr__(self, 'label', tmp_label)
            else:
                object.__setattr__(self, 'label', value)
        else:
            object.__setattr__(self, key, value) 

    def __len__(self):
        if self.data is None:
            return 0
        else:
            return self.data.shape[0]
    
    def __repr__(self):
        return self.attributes

    def __iter__(self):
        return self

    def __next__(self):
        if (self.idx > len(self)//self.batch and len(self) % self.batch != 0) or (self.idx >= len(self)//self.batch and len(self) % self.batch == 0):
            self.idx = 0
            self.shuffle()
            raise StopIteration()
        features = self.data[self.idx*self.batch:(self.idx+1)*self.batch]
        if self.label is not None:
            target = self.label[self.idx*self.batch:(self.idx+1)*self.batch]
            self.idx += 1
            return Variable(features, requires_grad=False), Variable(target, requires_grad=False) 
        self.idx += 1
        return Variable(features, requires_grad=False) 

    def shuffle(self):
        idx = np.random.permutation(len(self.data))
        self.data = self.data[idx]
        if self.label is not None:
            object.__setattr__(self, 'label', self.label[idx])

    def to_one_hot(self):
        if self.label is None:
            raise ValueError
        if np.issubdtype(self.label.dtype, np.float64):
            raise TypeError
        if self.corr is None:
            labels = collections.Counter(self.label)
            corr = {key:i for i, key in enumerate(sorted(labels.keys()))}
            object.__setattr__(self, 'corr', corr)
        if np.issubdtype(self.label.dtype, np.string_):
            tmp = list(map(lambda x:self.corr[x],self.label))
        else:
            tmp = self.label
        object.__setattr__(self, 'label', np.eye(len(self.corr),dtype=np.int32)[self.label])

    def to_vector(self):
        raise NotImplementedError      

class DataLoader(object):
    '''DataLoader\n 

    Attributes: 
        train_dataset (DataSet): training dataset that consists of numpy array for features, labels and its size 
        valid_dataset (DataSet): validation dataset that consists of numpy array for features, labels and its size 
        test_dataset (DataSet): testing dataset that consists of numpy array for features, labels and its size 
        batch (int): batch size for sampling
    ''' 
    def __init__(self):
        self.train_dataset = DataSet()
        self.valid_dataset = DataSet()
        self.test_dataset = DataSet()

    def set_batch(self, batch):
        self.train_dataset.batch = batch
        self.valid_dataset.batch = batch
        self.test_dataset.batch = batch
        
class ImageLoader(DataLoader):
    '''ImageLoader\n
    
    DataLoader class that carries data augmentation functions for images
    '''
    def __init__(self):
        super().__init__()

    def crop(self, num):
        ''' Random Crop
        '''
        pass

    def scale(self, num):
        ''' Scale Augmentation
        '''
        pass

    def rotate(self, num):
        pass

    def flip(self, num):
        pass
    
    def noise(self, num):
        pass
