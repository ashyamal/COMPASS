import os
import numpy as np
import random, torch
import matplotlib.pyplot as plt
import pandas as pd

from compass import PreTrainer, FineTuner, loadcompass


# Sample datasets provided in GitHub repo
#TODO: Replace with updated TCGA data (v43 processed ourselves)

tcga_train_sample = pd.read_csv('./data/tcga_example_train.tsv', sep='\t', index_col=0)
tcga_test_sample = pd.read_csv('./data/tcga_example_test.tsv', sep='\t', index_col=0)

# Define pretraining hyperparameters
pt_args = {'lr': 1e-3, 'batch_size': 96, 'epochs': 20, 'seed':42}
pretrainer = PreTrainer(**pt_args)

# Train the model using the provided training and test datasets
# - dfcx_train: Training dataset
# - dfcx_test: Validation dataset to monitor performance
pretrainer.train(dfcx_train=tcga_train_sample,
                 dfcx_test=tcga_test_sample)

# Save the trained pretrainer model for future use
pretrainer.save('./model/my_pretrainer.pt')