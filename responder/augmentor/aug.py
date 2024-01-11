import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.nn.functional import normalize
from torch.distributions import Beta

import torch
import numpy as np


class MixupNomralAugmentor:
    """Mixup from normal samples with selected genes"""

    def __init__(self, df_tpm_normal, genes2mixup=[], beta = 0.9, n_views=1):
        '''
        Selective mixup
        '''
        if len(genes2mixup) == 0:
            genes2mixup = df_tpm_normal.columns
            
        X = torch.tensor(df_tpm_normal.values,dtype=torch.float32).clone().detach()
        #X = normalize(torch.log2(X + 1.), p=2.0, dim = 0)
        
        self.genes2mixup = genes2mixup
        self.select_idx = df_tpm_normal.columns.get_indexer(genes2mixup)
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

        return x1.to(x.device)  

    def augment(self, x):
        return [self._transform(x) for i in range(self.n_views)]

    def __repr__(self):
        return self.__class__.__name__ + '(beta={}, n_views={})'.format(self.beta, self.n_views)





class RandomMaskAugmentor:
    """Random Mask Augmentation for Gene Expression Vectors"""

    def __init__(self, mask_probability = 0.1, mask_self=False, mask_self_probability = 0.2, n_views=1):
        '''
        Initialize the random mask augmentor with the given data and parameters.
        '''
        self.mask_probability = mask_probability
        self.mask_self = mask_self
        self.mask_self_probability = mask_self_probability
        self.n_views = n_views

        
    def _transform(self, x):
        
        if self.mask_self:
            # to mask the x itself
            x_self = x.numpy()
            mask = np.random.rand(*x_self.shape) < self.mask_self_probability
            x_self[mask] = 0

        x_np = x.clone().numpy()
        random_values = np.random.rand(*x_np.shape)
        mask = random_values < self.mask_probability
        x_np[mask] = 0
        x_new = torch.from_numpy(x_np).to(x.device)
        return x_new
        

    def augment(self, x):
        return [self._transform(x) for _ in range(self.n_views)]

    
    def __repr__(self):
        return self.__class__.__name__ + '(mask_probability={}, n_views={})'.format(self.mask_probability, 
                                                                                    self.n_views)


class FeatureJitterAugmentor:
    """Feature Jittering for Gene Expression Vectors"""

    def __init__(self, jitter_std = 0.01, n_views=1):
        '''
        Initialize the feature jitter augmentor with the given data and parameters.
        '''
        self.jitter_std = jitter_std
        self.n_views = n_views

    def _transform(self, x):
        # Generate jittering noise
        jitter = torch.normal(mean=0, std=self.jitter_std, size=x.size())

        # Apply jittering to the vector
        x_jittered = x.clone() + jitter

        return x_jittered.to(x.device)

    def augment(self, x):
        return [self._transform(x) for _ in range(self.n_views)]

    def __repr__(self):
        return self.__class__.__name__ + '(jitter_std={}, n_views={})'.format(self.jitter_std, self.n_views)
