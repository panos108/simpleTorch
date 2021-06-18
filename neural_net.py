import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, state_size, action_size, action_range= None):
        super().__init__()
        if action_range==None:
            # self.action_low, self.action_high =\
            #     torch.from_numpy(np.array([0.]*action_size)), torch.from_numpy(np.array([1.]*action_size))#, torch.from_numpy(np.array([1.]*action_size))
            self.range_available = False
        else:
            self.range_available = True
            self.action_low, self.action_high = torch.from_numpy(np.array(action_range))
        self.layer1 = nn.Linear(state_size, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 20)
        self.action = nn.Linear(20, action_size)

    def forward(self, state):
        m      = torch.nn.LeakyReLU(0.1)#0.01)
        layer1 = m(self.layer1(state))
        layer2 =m(self.layer2(layer1))
        layer3 = m(self.layer3(layer2))
        action = (self.action(layer3))
        if self.range_available:
            return self.action_low + (self.action_high - self.action_low) * (action)
        else:
            return (action)


class Model_probabilistic(nn.Module):
    def __init__(self, state_size, action_size, action_range= None):
        super().__init__()
        if action_range==None:
            # self.action_low, self.action_high =\
            #     torch.from_numpy(np.array([0.]*action_size)), torch.from_numpy(np.array([1.]*action_size))#, torch.from_numpy(np.array([1.]*action_size))
            self.range_available = False
        else:
            self.range_available = True
            self.action_low, self.action_high = torch.from_numpy(np.array(action_range))
        self.layer1 = nn.Linear(state_size, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 20)
        self.action = nn.Linear(20, action_size)
        self.action_std = nn.Linear(20, action_size)

    def forward(self, state):
        m      = torch.nn.LeakyReLU(0.1)#0.01)
        relu   = torch.nn.ReLU()
        layer1 = m(self.layer1(state))
        layer2 =m(self.layer2(layer1))
        layer3 = m(self.layer3(layer2))
        action_mean = (self.action(layer3))
        action_sts = relu(self.action_std(layer3))

        return action_mean, action_sts
