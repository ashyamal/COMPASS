from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as prc_auc_score
from sklearn.feature_selection import mutual_info_classif

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed



def _auc(y_true, y_prob, method = 'roc'):

    select = ~y_true.isna() & ~y_prob.isna()
    y_prob = y_prob[select]
    y_true = y_true[select]
    
    if len(y_true) == 0:
        return np.nan
    if method == 'roc':
        score = roc_auc_score(y_true, y_prob)
    elif method == 'prc':
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        score = prc_auc_score(recall, precision)
    else:
        score = mutual_info_classif(y_prob.values.reshape(-1,1), 
                                    y_true, random_state=123)
        score = score[0]
    return score

  
def _fuc(x):
    dfx, dfy, method, col = x
    score = _auc(y_true = dfy, y_prob = dfx, method=method)
    return {'gene_name':col,method:score}


def Xy_score(dfxy, y_col, method = 'prc', n_jobs=6):
    '''
    dfxy: dataframe with X and y columns
    y_col: y column name
    method: {'prc', 'roc', 'mic'}
    '''

    x_col = dfxy.columns[~dfxy.columns.isin([y_col])]
    dfX = dfxy[x_col]
    dfy = dfxy[y_col].map({'R':1, 'NR':0})
    combination_list = [(dfX[col], dfy, method, col) for col in x_col]
    P = Parallel(n_jobs=n_jobs)
    res = P(delayed(_fuc)(x) for x in tqdm(combination_list, ascii=True)) 
    score = pd.DataFrame(res).set_index('gene_name')
    return score
    













