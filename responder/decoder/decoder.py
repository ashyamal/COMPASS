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





class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(SoftmaxClassifier, self).__init__()
        # We only define the shape of W and b here without initializing them
        self.W = nn.Parameter(torch.Tensor(out_dim, input_dim))  # Will be initialized later with class means
        self.b = nn.Parameter(torch.zeros(out_dim))  # Bias initialized as zeros
        self.num_classes = out_dim
        self.feature_dim = input_dim

    
    def forward(self, x):
        # Normalize the input features and weights to calculate cosine similarity
        x_norm = F.normalize(x, p=2, dim=1)
        W_norm = F.normalize(self.W, p=2, dim=1)
        # Calculate the cosine similarity between the input and prototypes
        # Using torch.mm for batch matrix multiplication
        cosine_similarity = torch.mm(x_norm, W_norm.T)
        # Add the bias term
        logits = cosine_similarity + self.b
        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=1)
        return probabilities

    def initialize_parameters(self, support_features, support_labels):
        """
        Initialize the weights (W) using the class means of the support set features and the biases (b) as zeros.
    
        Parameters:
        support_features (torch.Tensor): The features of the support set.
        support_labels (torch.Tensor): The one-hot encoded labels of the support set.
        """
        with torch.no_grad():  # No need to track gradients during initialization
            for i in range(self.num_classes):
                # Calculate the mean feature vector for class i
                class_features = support_features[support_labels[:, i] == 1]
                class_mean = class_features.mean(dim=0)
                # Assign the mean vector to the corresponding row in W
                self.W[i] = class_mean


