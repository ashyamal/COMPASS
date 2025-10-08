import os
import pandas as pd
import wandb
from compass import PreTrainer
import numpy as np

# Load data
# TODO: replace with processed TCGA data from v43
# Current train set: 495 samples x 1066 genes
# Current test set: 99 samples x 1066 genes
tcga_train_sample = pd.read_csv('../example/data/tcga_example_train.tsv', sep='\t', index_col=0)
tcga_test_sample = pd.read_csv('../example/data/tcga_example_test.tsv', sep='\t', index_col=0)

def train():
    
    run = wandb.init()
    config = wandb.config
    
    print(f"\n{'='*70}")
    print(f"Training: {config.encoder} | LR: {config.lr} | Batch: {config.batch_size}")
    print(f"{'='*70}\n")
    
    pt_args = {
        'encoder': config.encoder,
        'lr': config.lr,
        'batch_size': config.batch_size,
        'epochs': config.epochs,
        'seed': config.seed,
        'transformer_dim': 32,
        'transformer_num_layers': 2,
        'transformer_nhead': 2,
        'encoder_dropout': 0.2,
        'with_wandb': False,
    }
    
    pretrainer = PreTrainer(**pt_args)
    pretrainer.train(dfcx_train=tcga_train_sample, dfcx_test=tcga_test_sample)
    
    # Log all epochs for training curves
    for epoch_data in pretrainer.performace:
        epoch, train_total, train_ssl, train_task, test_total, test_ssl, test_task = epoch_data
        wandb.log({
            'train_loss': train_total,
            'test_loss': test_total,
            'train_ssl_loss': train_ssl,
            'test_ssl_loss': test_ssl,
        })
    
    # Log final summary
    final_test_loss = pretrainer.performace[-1][4]
    best_test_loss = min([p[4] for p in pretrainer.performace if not np.isnan(p[4])])
    
    wandb.summary['final_train_loss'] = pretrainer.performace[-1][1]
    wandb.summary['final_test_loss'] = final_test_loss
    wandb.summary['best_test_loss'] = best_test_loss
    
    # Save model
    model_dir = f'./models/{config.encoder}'
    os.makedirs(model_dir, exist_ok=True)
    model_path = f'{model_dir}/lr{config.lr}_bs{config.batch_size}.pt'
    pretrainer.save(model_path)
    
    print(f"Complete: Test Loss = {final_test_loss:.4f}")

# Define sweep configuration
sweep_config = {
    'method': 'grid',  # Try all combinations
    'name': 'identity-vs-transformer',
    'metric': {
        'name': 'best_test_loss',  # Metric to optimize
        'goal': 'minimize'
    },
    'parameters': {
        'encoder': {
            'values': ['identity', 'transformer']
        },
        'lr': {
            'values': [5e-4, 1e-3, 5e-3]  # 3 learning rates
        },
        'batch_size': {
            'values': [64, 96, 128]  # 3 batch sizes
        },
        'seed': {
            'value': 42  # Fixed
        },
        'epochs': {
            'value': 20  # Fixed
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project='COMPASS-pretrain-transformer-ablation')

print("="*70)
print("WANDB SWEEP CONFIGURATION")
print("="*70)
print(f"Project: compass-ablation")
print(f"Sweep ID: {sweep_id}")
print(f"\nHyperparameters:")
print(f"  Encoders: identity, transformer")
print(f"  Learning rates: 5e-4, 1e-3, 5e-3")
print(f"  Batch sizes: 64, 96, 128")
print(f"  Epochs: 20")
print(f"\nTotal experiments: 2 x 3 x 3 = 18 runs")
print("="*70)
print("\nStarting sweep agent...\n")

# Run sweep
wandb.agent(sweep_id, function=train, count=18)

print("\n" + "="*70)
print("SWEEP COMPLETE!")
print("="*70)