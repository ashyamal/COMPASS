from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as prc_auc_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

def score(y_true, y_prob, y_pred):
    
    select = ~y_true.isna() 
    y_prob = y_prob[select]
    y_true = y_true[select]
    y_pred = y_pred[select] #.map({'NR':0, 'R':1})    

    if len(y_true.unique()) == 1:
        roc = np.nan
        prc = np.nan
    else:
        roc = roc_auc_score(y_true, y_prob)
        _precision, _recall, _ = precision_recall_curve(y_true, y_prob)
        prc = prc_auc_score(_recall, _precision)
        
    f1 = f1_score(y_true, y_pred, pos_label=1)
    acc = accuracy_score(y_true, y_pred)

    return roc, prc, f1, acc



