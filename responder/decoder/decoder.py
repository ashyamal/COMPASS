import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ModuleList, ReLU, Dropout
from copy import deepcopy 



class ClassDecoder(nn.Module):
    def __init__(self, input_dim=32, dense_layers = [], out_dim = 1, dropout_p = 0.0, batch_norms = False):
        '''
        classification
        '''
        super(ClassDecoder,self).__init__()

        _dense_layers = [input_dim]
        _dense_layers.extend(dense_layers)
        self._dense_layers = _dense_layers
        self.batch_norms = batch_norms
        
        ## Dense
        self.lins = ModuleList()
        for i in range(len(_dense_layers)-1):
            lin = Linear(_dense_layers[i], _dense_layers[i+1])
            self.lins.append(lin)

        ## Batchnorm
        self._batch_norms = ModuleList()
        for i in range(len(_dense_layers)-1):
            self._batch_norms.append(deepcopy(torch.nn.BatchNorm1d(_dense_layers[i+1])))

            
        ## Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Output layer
        last_hidden = _dense_layers[-1]
        self.out = Linear(last_hidden, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for lin, norm in zip(self.lins, self._batch_norms):
            if self.batch_norms:
                x = self.dropout(F.relu(norm(lin(x)), inplace=True))
            else:
                x = self.dropout(F.relu(lin(x), inplace=True))                
        y = self.softmax(self.out(x))
        #y = F.log_softmax(self.out(x), dim=1)
        return y
        

class RegDecoder(nn.Module):
    def __init__(self, input_dim=32, dense_layers = [], out_dim = 1, dropout_p = 0.0, batch_norms = False):
        '''
        Regression
        '''
        super(RegDecoder,self).__init__()

        _dense_layers = [input_dim]
        _dense_layers.extend(dense_layers)
        self._dense_layers = _dense_layers
        self.batch_norms = batch_norms
        
        ## Dense
        self.lins = ModuleList()
        for i in range(len(_dense_layers)-1):
            lin = Linear(_dense_layers[i], _dense_layers[i+1])
            self.lins.append(lin)

        ## Batchnorm
        self._batch_norms = ModuleList()
        for i in range(len(_dense_layers)-1):
            self._batch_norms.append(deepcopy(torch.nn.BatchNorm1d(_dense_layers[i+1])))

        ## Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Output layer
        last_hidden = _dense_layers[-1]
        self.out = Linear(last_hidden, out_dim)


    def forward(self, x):
        for lin, norm in zip(self.lins, self._batch_norms):
            if self.batch_norms:
                x = self.dropout(F.relu(norm(lin(x)), inplace=True))
            else:
                x = self.dropout(F.relu(lin(x), inplace=True))                
        y = self.out(x)
        return y

