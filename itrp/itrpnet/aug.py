import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.nn.functional import normalize
from torch.distributions import Beta

class MixupNomralAugmentor:
    """Mixup from normal samples with selected genes"""

    def __init__(self, df_tpm_normal, select_genes, beta = 0.9, n_views=1):
        '''
        Selective mixup
        '''
        X = torch.tensor(df_tpm_normal.values,dtype=torch.float32).clone().detach()
        X = normalize(torch.log2(X + 1.), p=2.0, dim = 0)
        self.select_genes = select_genes
        self.select_idx = df_tpm_normal.columns.get_indexer(select_genes)
        self.X_mix = X[:, self.select_idx]

        self.n = len(df_tpm_normal)
        self.beta = beta
        self.p =  Beta(torch.FloatTensor([beta]), torch.FloatTensor([beta]))
        self.n_views = n_views

    ## remember to do a select mixup
    def _transform(self, x):
        #b1 = self.p.sample()
        b1 = self.beta
        m1 = self.X_mix[np.random.choice(self.n)]
        x1 = x.clone().detach()
        xs = x1[self.select_idx]
        xs = b1* xs + (1-b1) * m1
        x1[self.select_idx] = xs
        return x1  

    def augment(self, x):
        return [self._transform(x) for i in range(self.n_views)]

    def __repr__(self):
        return self.__class__.__name__ + '(beta={}, n_views={})'.format(self.beta, self.n_views)