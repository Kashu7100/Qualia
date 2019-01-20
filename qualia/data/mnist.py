
# -*- coding: utf-8 -*- 
import os
import gzip 
import numpy as np 
from ..util import ImageLoader 
from ..autograd import Variable 

class MNIST(ImageLoader): 
    '''MNIST Dataset\n 
    This will load the MNIST dataset as Valiable object. 
    The MNIST dataset coonsists of handwritten digits with a training set of 60,000 examples and a test set of 10,000 examples. 
    For the further information, visit MNIST database as link below: 
 
    http://yann.lecun.com/exdb/mnist/ 
 
    Args: 
        batch (int): Batch size used when loading data
        normalize (bool): If true, the intensity value of a specific pixel in a specific image will be rescaled from [0, 255] to [0, 1]. Default: True 
        flatten (bool): If true, data will have a shape of [N, 28*28]. Default: False 
        one_hot_label (bool): If true, labels will be stored in one hot expression. Default: True 
     
    Shape: 
        default: [N, 1, 28, 28] 
        flatten: [N, 28*28] 
 
    Example: 
        >>> mnist = MNIST() 
        >>> # load data and labels with specified batch number 
        >>> for data, label in mnist.train_data(): 
        >>>     y = model(data) 
        >>>     loss = criteria(y, label) 
        >>>     loss.backward() 
        >>>     ... 
    ''' 
    def __init__(self, normalize=True, flatten=False, one_hot_label=True): 
        super().__init__() 
        self.path = os.path.dirname(os.path.abspath(__file__)) 

        print('[*] preparing data...')
        print('    this might take few minutes.') 
        if not os.path.exists(self.path + '/mnist/'): 
            os.makedirs(self.path + '/mnist/') 
            self.download(self.path+'/mnist/')

        self.train_dataset.data = self._load_data(self.path + '/mnist/train_data.gz')
        self.train_dataset.label = self._load_label(self.path + '/mnist/train_labels.gz')     
        self.train_dataset.attributes = 'MNIST train detaset contains 60000 examples'
        self.test_dataset.data = self._load_data(self.path + '/mnist/test_data.gz') 
        self.test_dataset.label = self._load_label(self.path + '/mnist/test_labels.gz') 
        self.test_dataset.attributes = 'MNIST test dataset contains 10000 examples'

        print('[*] done.') 

        if normalize: 
            self.train_dataset.data = self.train_dataset.data / 255.0
            self.test_dataset.data = self.test_dataset.data / 255.0 
        if flatten:
            self.train_dataset.data = self.train_dataset.data.reshape(-1, 28*28) 
            self.test_dataset.data = self.test_dataset.data.reshape(-1, 28*28) 
        if one_hot_label: 
            self.train_dataset.to_one_hot()
            self.test_dataset.to_one_hot()
 
    def download(self, path): 
        import urllib.request 
        url = 'http://yann.lecun.com/exdb/mnist/' 
        files = { 
            'train_data.gz':'train-images-idx3-ubyte.gz', 
            'train_labels.gz':'train-labels-idx1-ubyte.gz', 
            'test_data.gz':'t10k-images-idx3-ubyte.gz', 
            'test_labels.gz':'t10k-labels-idx1-ubyte.gz' 
        } 
        for key, value in files.items(): 
            if not os.path.exists(path+key): 
                urllib.request.urlretrieve(url+value, path+key) 
 
    def _load_data(self, filename):
        with gzip.open(filename, 'rb') as file: 
            data = np.frombuffer(file.read(), np.uint8, offset=16) 
        return data.reshape(-1,1,28,28) 

    def _load_label(self, filename):
        with gzip.open(filename, 'rb') as file: 
            labels = np.frombuffer(file.read(), np.uint8, offset=8) 
        return labels
