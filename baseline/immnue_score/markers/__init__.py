# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:44:09 2023

@author: Wanxiang Shen
"""


import pandas as pd
import os

cwd = os.path.dirname(__file__)
MARKERS = pd.read_csv(os.path.join(cwd, 'marker.tsv'), sep='\t', index_col=0)
