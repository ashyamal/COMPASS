import wandb
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ModuleList, ReLU, Dropout
from copy import deepcopy 
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score,roc_curve
import torch.optim as optim



class MSIPredictor(nn.Module):
    def __init__(self, input_dim=32, dense_layers = [], out_dim = 3, dropout_p = 0.0):
        '''
        MSI classification
        '''
        super(MSIPredictor,self).__init__()

        _dense_layers = [input_dim]
        _dense_layers.extend(dense_layers)
        self._dense_layers = _dense_layers

        ## Dense
        self.lins = ModuleList()
        for i in range(len(_dense_layers)-1):
            lin = Linear(_dense_layers[i], _dense_layers[i+1])
            self.lins.append(lin)

        ## Batchnorm
        self.batch_norms = ModuleList()
        for i in range(len(_dense_layers)-1):
            self.batch_norms.append(deepcopy(torch.nn.BatchNorm1d(_dense_layers[i+1])))

        ## Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Output layer
        last_hidden = _dense_layers[-1]
        self.out = Linear(last_hidden, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for lin, norm in zip(self.lins, self.batch_norms):
            x = self.dropout(F.relu(lin(x), inplace=True))
        y = self.softmax(self.out(x))
        return y

class TMBPredictor(nn.Module):
    def __init__(self, input_dim=32, dense_layers = [], out_dim = 1, dropout_p = 0.0):
        '''
        TMB prediction
        '''
        super(TMBPredictor,self).__init__()

        _dense_layers = [input_dim]
        _dense_layers.extend(dense_layers)
        self._dense_layers = _dense_layers

        ## Dense
        self.lins = ModuleList()
        for i in range(len(_dense_layers)-1):
            lin = Linear(_dense_layers[i], _dense_layers[i+1])
            self.lins.append(lin)

        ## Batchnorm
        self.batch_norms = ModuleList()
        for i in range(len(_dense_layers)-1):
            self.batch_norms.append(deepcopy(torch.nn.BatchNorm1d(_dense_layers[i+1])))

        ## Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Output layer
        last_hidden = _dense_layers[-1]
        self.out = Linear(last_hidden, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for lin, norm in zip(self.lins, self.batch_norms):
            #x = self.dropout(F.relu(norm(lin(x)), inplace=True))
            x = self.dropout(F.relu(lin(x), inplace=True))
        y = self.out(x)
        return y


class CNVPredictor(nn.Module):
    def __init__(self, input_dim=32, dense_layers = [], out_dim = 1, dropout_p = 0.0):
        '''
        CNV prediction
        '''
        super(CNVPredictor,self).__init__()

        _dense_layers = [input_dim]
        _dense_layers.extend(dense_layers)
        self._dense_layers = _dense_layers

        ## Dense
        self.lins = ModuleList()
        for i in range(len(_dense_layers)-1):
            lin = Linear(_dense_layers[i], _dense_layers[i+1])
            self.lins.append(lin)

        ## Batchnorm
        self.batch_norms = ModuleList()
        for i in range(len(_dense_layers)-1):
            self.batch_norms.append(deepcopy(torch.nn.BatchNorm1d(_dense_layers[i+1])))

        ## Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Output layer
        last_hidden = _dense_layers[-1]
        self.out = Linear(last_hidden, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for lin, norm in zip(self.lins, self.batch_norms):
            #x = self.dropout(F.relu(norm(lin(x)), inplace=True))
            x = self.dropout(F.relu(lin(x), inplace=True))
        y = self.out(x)
        return y



# Define your model architecture here
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, emb):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, emb),
                                    nn.BatchNorm1d(emb),
                                    nn.ReLU(inplace=True),
          )
        # self.tt = nn.TransformerEncoderLayer(emb, 4)
        # self.encoder = nn.TransformerEncoder(self.tt, 1)
    def forward(self,x):
        x = self.encoder(x)
        # x, _ = torch.max(output, dim=1)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, emb):
        super().__init__()
        self.mlp = MLPEncoder(input_dim, emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb, nhead=4, dim_feedforward = 256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = 6)
    def forward(self,x):
        x = self.encoder(self.mlp(x))
        # x, _ = torch.max(output, dim=1)
        return x
        


class TCGAPretrainModel(nn.Module):
    def __init__(self, input_dim, middle_emb_dim = 512, out_emb_dim = 32):
        super().__init__()
        self.backbone = MLPEncoder(input_dim, emb=middle_emb_dim) #
        self.projection = nn.Sequential(nn.Linear(in_features=middle_emb_dim, out_features=out_emb_dim),)
        self.tmb_predictor = TMBPredictor(out_emb_dim, dense_layers = [16], out_dim = 1)
        self.msi_predictor = MSIPredictor(out_emb_dim, dense_layers = [16], out_dim = 3)
        self.cnv_predictor = CNVPredictor(out_emb_dim, dense_layers = [16], out_dim = 1)
        

        # model_args = {'input_dim':input_dim, 
        #             'middle_emb_dim':middle_emb_dim,
        #             'out_emb_dim':out_emb_dim, }

        # self.model_args = model_args

    
    def forward(self, x):
        embedding = self.backbone(x)
        project = self.projection(embedding)
        emb_vector = F.normalize(project, p=2., dim=1)
        
        y_tmb = self.tmb_predictor(emb_vector)
        y_msi = self.msi_predictor(emb_vector)
        y_cnv = self.cnv_predictor(emb_vector)
        
        return emb_vector, (y_tmb, y_msi, y_cnv)  






class classifier(nn.Module):
    def __init__(self,input_dim=32):
        super(classifier,self).__init__()
        self.fc1 = nn.Linear(input_dim,2)
        self.bn1 = nn.BatchNorm1d(2)
        self.fc2 = nn.Linear(32,2)
        self.softmax = nn.Softmax(dim=1)
        self.do1 = nn.Dropout(0.3)  

    def forward(self,x):
        x = F.relu(self.bn1(self.fc1(x)))
        # x = self.fc2(x)
        # x = self.do1(self.fc1(x))
        # return self.softmax(x)
        return F.log_softmax(x, dim=1)

