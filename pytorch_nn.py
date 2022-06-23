# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:29:22 2022

@author: oe21s024
"""

import torch.nn as nn # this gives access to layers of neural network
import torch.nn.functional as F # this gives access to functions such as sigmoid and relu
import torch.optim as optim # this gives access to optimizers 
import torch as T # base package - torch

# linear classifier class
class LinearClassifier(nn.module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128) # linear classification layer ---> not a convolution layer
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,n_classes)
        
        self.optimizer = optim.Adam(self.parameters, lr = lr)
        self.loss = nn.CrossEntropyLoss() # we generally use mean square loss our problem
        # but this is a simple linear classification problem
        
        # if you have gpu how to use it 
        # next we need to send the network to the device to make it work
        self.device  = T.device('cudo:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data))
        layer2 = F.sigmoid(self.fc2(layer1))
        layer3 = self.fc3(layer2) # here at the output we wont do any activation
        
        return layer3 
    
   
    # next we are going to deal with learning loop
    # the learning function for the q network will be slightly different 
    # it takes in state, reward, action, next state
    # but in the neural network the learning function needs data and labels only
    def learn(self, data, labels): # classification problem, so label is also an input 
        '''takes in data and labels and
        tell us which kind of label does the data belong to
        '''
        self.optimizer.zero_grad()
        data = T.tensor(data).to(self.device)
        labels = T.tensor(labels).to(self.labels)
        
        predictions = self.forward(data)       # feed forwarding to get predictions

        
        cost =  self.loss(predictions, labels) # finding loss by comparing predictions and labels 
        
        
        cost.backward()   # back propogating
        self.optimzer.step() # optimizing the results 
        
        
        
        
        
        
        