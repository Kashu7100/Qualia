from qualia.nn.modules import Module, Linear
from qualia.nn.functions import relu
from qualia.optim import *
from qualia.util import ReplayMemory

import gym

class NeuralNet(Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.linear1 = Linear(in_features, hidden)
        self.linear2 = Linear(hidden, hidden)
        self.linear3 = Linear(hidden, out_features)
        
    def forward(self, x):
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = self.linear3(x)
        return x
      
class DDQN(object):
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.main_q_network = Net(num_states, 32, num_actions)
        self.target_q_network = Net(num_states, 32, num_actions)
        
        self.memory = ReplayMemory(10000)
        
        self.optim = Adam(self.main_q_network.params)
        
    def __call__(self, , state, episode):
        pass
