# -*- coding: utf-8 -*-
"""

Calculate the feature combinations
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

def _yield_combinations(dfdata, opt):
   
    cols = dfdata.columns
    data = dfdata.values
    N = len(cols)
    for i1 in range(N):
        for i2 in range(i1):
            n1 = cols[i1]
            n2 = cols[i2]
            s1 = data[:, i1]
            s2 = data[:, i2]
            yield (s1, s2, opt, n1, n2)

def _fuc(x):
    s1, s2, opt, n1, n2 = x
    if opt == ' - ':
        s = s1 - s2
    else:
        s = s1 + s2
    s = pd.Series(s)
    s.name = '%s%s%s' % (n1, opt, n2)
    return s
    

class pairwise_combination:

    def __init__(self, combination_pairs = [], n_cpus=8, method='subtraction'):
        """
        parameters
        ---------------
        combination_pairs: [(node1, node2)], the node should be the column name of dfdata, 
                           if no pairs provided, it will calculate (n*(n-1)/2) possibale pairs
        method: {'subtraction', 'addition'}
        n_cpus: n_cpus    
    
        Usage
        --------------
        >>> import numpy as np
        >>> import pandas as pd
        >>> data = np.random.random_sample(size=(10000,10)
        >>> dfdata = pd.DataFrame(data)
     
        """   
        self.combination_pairs = combination_pairs
        self.n_cpus = n_cpus
        self.method = method
        self.method_opt =  {'subtraction':' - ', 'addition':' + '}
        self.opt = self.method_opt[method]
        
        
    def fit(self, dfdata):
        '''
        dfdata: pd.DataFrame, columns--> genes|features|nodes, rows--> samples|patients
        '''
        self.index = dfdata.index
        if len(self.combination_pairs) != 0:
            combination_list = []
            for (n1, n2) in self.combination_pairs:
                s1 = dfdata[n1]
                s2 = dfdata[n2]
                element = (s1, s2, self.opt, n1, n2)
                combination_list.append(element)
        else:
            combination_list = list(_yield_combinations(dfdata, self.opt))
        self.combination_list = combination_list
        return self

    def transform(self):
        '''
        transform combined features
        '''
        #combination_list = [(pair[0], pair[1], opt) for pair in combination_pairs]
        P = Parallel(n_jobs=self.n_cpus)
        res = P(delayed(_fuc)(x) for x in tqdm(self.combination_list, ascii=True)) 
        df = pd.concat(res, axis=1)
        df.index = self.index
        return df


    def inverse(self, df):
        names = []
        for col in df.columns:
            a, b = col.split(self.opt)
            name = '%s%s%s' % (b, self.opt, a)
            names.append(name)
        df_inverse = -df
        df_inverse.columns  = names        
        return df_inverse


def pairwise_combination2(df, chunk_size = 5000000):
    '''
    df: pd.DataFrame, columns--> genes|features|nodes, rows--> samples|patients
    '''
    
    gene_values = df.values
    # 计算基因两两相互减并生成新的组合矩阵
    combinations = [(gene1, gene2) for gene1 in df.columns for gene2 in df.columns if gene1 != gene2]
    gene_combinations_list = [combinations[i:i + chunk_size] for i in range(0, len(combinations), chunk_size)] 
    for i, gene_combinations in enumerate(gene_combinations_list):
        combination_features = {}
        for gene1, gene2 in tqdm(gene_combinations, ascii=True):
            combination_feature = gene_values[:, df.columns.get_loc(gene1)] - gene_values[:, df.columns.get_loc(gene2)]
            combination_features[f'{gene1} - {gene2}'] = combination_feature
        # 创建包含结果的DataFrame
        result_df = pd.DataFrame(combination_features, indexx=df.index)
        del combination_features
        yield result_df


        


    