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

    def __init__(self, df_tpm_normal, genes2mixup=[], 
                 mix_beta = 0.9, n_views=1):
        '''
        Selective mixup
        '''
    
        if len(genes2mixup) == 0:
            genes2mixup = df_tpm_normal.columns
            
        X = torch.tensor(df_tpm_normal.values,dtype=torch.float32).clone().detach()
        
        self.genes2mixup = genes2mixup
        self.select_idx = df_tpm_normal.columns.get_indexer(genes2mixup)
        self.X_mix = X[:, self.select_idx]

        self.n = len(df_tpm_normal)
        self.beta = mix_beta
        self.p =  Beta(torch.FloatTensor([mix_beta]), torch.FloatTensor([mix_beta]))
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

    def __init__(self, 
                 mask_p_prob = 0.5,
                 mask_a_prob = None, 
                 mask_n_prob = None,
                 n_views=1):
        '''
        Initialize the random mask augmentor with the given data and parameters.
        '''
        if mask_a_prob is None:
            mask_a_prob = mask_p_prob

        if mask_n_prob is None:
            mask_n_prob = 0.0
            
        self.mask_p_prob = mask_p_prob
        self.mask_a_prob = mask_a_prob
        self.mask_n_prob = mask_n_prob
        self.n_views = n_views

        
    def _transform(self, x, probability):
        
        x_np = x.clone().numpy()
        random_values = np.random.rand(*x_np.shape)
        mask = random_values < probability
        x_np[mask] = 0
        x_new = torch.from_numpy(x_np).to(x.device)
        return x_new

    def _augment(self, x, mask_prob):
        return [self._transform(x, mask_prob) for _ in range(self.n_views)]

    def augment_p(self, p):
        return  self._augment(p, self.mask_p_prob)

    def augment_a(self, a):
        return  self._augment(a, self.mask_a_prob)

    def augment_n(self, n):
        return  self._augment(n, self.mask_n_prob)


    def __repr__(self):
        return self.__class__.__name__ + '(mask_probability=(a:{},p:{},n:{}), n_views={})'.format(
                                                                                    self.mask_a_prob,
                                                                                    self.mask_p_prob,
                                                                                    self.mask_n_prob,
                                                                                    self.n_views)


class FeatureJitterAugmentor:
    """Feature Jittering for Gene Expression Vectors"""

    def __init__(self, 
                 jitter_p_std = 0.2,
                 jitter_a_std = None, 
                 jitter_n_std = None,
                 n_views=1):
        '''
        Initialize the feature jitter augmentor with the given data and parameters.
        '''
        if jitter_a_std is None:
            jitter_a_std = jitter_p_std
        if jitter_n_std is None:
            jitter_n_std = 0.0
            
        self.n_views = n_views
        self.jitter_p_std = jitter_p_std
        self.jitter_a_std = jitter_a_std
        self.jitter_n_std = jitter_n_std

    
    def _transform(self, x, jitter_std):
        # Generate jittering noise
        jitter = torch.normal(mean=0, std=jitter_std, size=x.size())
        # Apply jittering to the vector
        x_jittered = x.clone() + jitter
        return x_jittered.to(x.device)

    def _augment(self, x, jitter_std):
        return [self._transform(x, jitter_std) for _ in range(self.n_views)]

    def augment_p(self, p):
        return  self._augment(p, self.jitter_p_std)

    def augment_a(self, a):
        return  self._augment(a, self.jitter_a_std)

    def augment_n(self, n):
        return  self._augment(n, self.jitter_n_std)

    
    def __repr__(self):
        return self.__class__.__name__ + '(jitter_std=(a:{},p:{},n:{}), n_views={})'.format(
                                                                                    self.jitter_a_std,
                                                                                    self.jitter_p_std,
                                                                                    self.jitter_n_std,
                                                                                    self.n_views)