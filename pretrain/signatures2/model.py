import wandb
import torch, rtdl
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn import Sequential, Linear, ModuleList, ReLU, Dropout
import torch.optim as optim
from torchvision.ops import MLP

from einops import rearrange, repeat
from copy import deepcopy 
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score,roc_curve
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np





class ClassMapper(nn.Module):
    def __init__(self, input_dim=32, dense_layers = [], out_dim = 1, dropout_p = 0.0, batch_norms = False):
        '''
        classification
        '''
        super(ClassMapper,self).__init__()

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
        return y
        

class RegMapper(nn.Module):
    def __init__(self, input_dim=32, dense_layers = [], out_dim = 1, dropout_p = 0.0, batch_norms = False):
        '''
        Regression
        '''
        super(RegMapper,self).__init__()

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


        
# Define your model architecture here
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, dense_layers = [512, 256, 128]):
        super().__init__()
        self.encoder = MLP(in_channels=input_dim, 
                           hidden_channels=dense_layers, 
                           norm_layer = torch.nn.BatchNorm1d, dropout  = 0.)
    def forward(self,x):
        x = self.encoder(x)
        return x




class GeneEmbedder(nn.Module):
    def __init__(self, num_genes, emb_dim, with_bias = False):
        '''
        embedding genes
        '''
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_genes, emb_dim))
        self.biases = nn.Parameter(torch.randn(num_genes, emb_dim))
        self.with_bias = with_bias
        
    def forward(self, x):
        x = x.unsqueeze(-1)        
        if self.with_bias:
            return x * self.weights + self.biases
        else:
            return x * self.weights

class NumEmbeddings(nn.Module):
    def __init__(self, n_features: int, d_embedding: int) -> None:
        super().__init__()
        FT = rtdl.NumericalFeatureTokenizer(n_features, d_embedding, True, 'uniform')
        layers = [FT, nn.ReLU()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class GlobalPool(nn.Module):
    def __init__(self,  reduce = 'mean'):
        '''
        reduce: {'mean', 'max', 'cls'}
        '''
        super(GlobalPool, self).__init__()
        self.reduce = reduce
        
    def forward(self, x):
        if self.reduce == 'mean':
            x = torch.mean(x, dim=1)
        elif self.reduce == 'max':
            x = torch.max(x, dim=1)
        else:
            x = x[:, 0] #cls token
        return x

        

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model = 512, num_layers = 2):
        super().__init__()
        
        self.input_linear = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, 
                                                        dim_feedforward = 128, 
                                                        dropout  = 0., 
                                                        batch_first = True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers)
     
    def forward(self,x):
        x = self.input_linear(x)
        x = x.unsqueeze(1)  # Add a sequence dimension (batch_size, seq_len, d_model)
        x = self.encoder(x)[:, 0, :]
        return x




class TransformerEncoder2(nn.Module):
    def __init__(self, input_dim, d_model = 128, num_layers = 2):
        super().__init__()

        self.embedder = GeneEmbedder(num_genes = input_dim, emb_dim = d_model, with_bias = True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, 
                                                        dim_feedforward = 128, 
                                                        dropout  = 0., batch_first = True)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
        self.pooler = GlobalPool(reduce = 'cls')

    
    def forward(self,x):
        x = self.pooler(self.encoder(self.embedder(x)))
        return x
        

class FTTransformer(nn.Module):
    def __init__(self, input_dim, d_model = 128, num_layers = 2):
        super().__init__()
        self.encoder = rtdl.FTTransformer.make_default(
            n_num_features=input_dim,
            cat_cardinalities=None,
            n_blocks = num_layers,
            last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
            d_out=d_model,)
        
    def forward(self,x):
        x = self.encoder(x, x_cat=None)
        return x



class TCGAPretrainModel(nn.Module):
    def __init__(self, 
                 input_dim, 
                 task_dim,
                 task_type,
                 embed_dim = 32, 
                 encoder = 'mlp', 
                 mlp_dense_layers = [512, 256, 128],
                 task_dense_layer=[16],
                 task_batch_norms = False,
                 transformer_dim = 128,
                 transformer_num_layers = 2,
                ):
        
        '''
        input_dim: input dim
        task_dim: supervised learning task dim
        task_type: {'r', 'c'}
        embed_dim: latent vector dim
        encoder: {'mlp', 'transformer'}
        task_dense_layer: dense layer of task
        '''
        super().__init__()

        self.input_dim = input_dim
        self.task_dim = task_dim
        self.task_type = task_type        
        self.embed_dim = embed_dim
        self.encoder = encoder
        self.mlp_dense_layers = mlp_dense_layers
        self.transformer_dim = transformer_dim
        self.transformer_num_layers = transformer_num_layers
        self.task_batch_norms = task_batch_norms
        self.task_dense_layer = task_dense_layer


        model_args = {'input_dim':self.input_dim, 
                'task_dim':self.task_dim,
                'task_type':self.task_type, 
                'embed_dim': self.embed_dim, 
                'encoder':self.encoder, 
                'mlp_dense_layers':self.mlp_dense_layers,
                'transformer_dim':self.transformer_dim,
                'transformer_num_layers':self.transformer_num_layers,
                'task_batch_norms':self.task_batch_norms,
                'task_dense_layer':self.task_dense_layer,                      
               }

        self.model_args = model_args

        
        if encoder == 'mlp':
            self._encoder = MLPEncoder(input_dim, dense_layers=mlp_dense_layers) #
            project_input_idm = mlp_dense_layers[-1]
        else:
            self._encoder = TransformerEncoder(input_dim, d_model = transformer_dim, 
                                              num_layers = transformer_num_layers) #
            project_input_idm = transformer_dim

        self._projector = nn.Sequential(nn.Linear(in_features=project_input_idm, 
                                                  out_features=embed_dim),)

        ## regression task
        if task_type == 'r':
            self.taskmapper = RegMapper(embed_dim, 
                                        dense_layers = task_dense_layer, 
                                        out_dim = task_dim, 
                                        batch_norms = task_batch_norms)
        
        ## classification task
        else:
            self.taskmapper = ClassMapper(embed_dim,
                                          dense_layers = task_dense_layer, 
                                          out_dim = task_dim, 
                                          batch_norms = task_batch_norms)


    def forward(self, x):
        encoding = self._encoder(x)
        projection = self._projector(encoding)
        embedding = F.normalize(projection, p=2., dim=1)
        y = self.taskmapper(embedding)
        return embedding, y






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

