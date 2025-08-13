#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======= Minimal importable version with overridable is_TPM_format & all_gene_mean_expression =======

import os
import sys
import types
import numpy as np
import random
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# --- Reproducibility same as original ---
def _set_seeds():
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    os.environ['PYTHONHASHSEED'] = str(123)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Globals preserved exactly as original script expects (defaults) ---
hidden_size = 200
num_layers = 2
num_heads = 4
dropout = 0.2
is_TPM_format = True
all_gene_mean_expression = 4992.7214  # used for data correction

# ---------------- Dataset (unchanged) ----------------
class MultitaskDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labelsA, labelsB):
        self.inputs = inputs
        self.labelsA = labelsA
        self.labelsB = labelsB

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labelsA = self.labelsA[index]
        labelsB = self.labelsB[index]
        return inputs, labelsA, labelsB

# ---------------- Model (unchanged math) ----------------
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(TransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        residual = x
        x, _ = self.multi_head_attention(x, x, x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_norm1(x + residual)
        residual = x
        x = self.feedforward(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_norm2(x + residual)
        return x

class GeneTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, dropout, is_TPM_format_ignored):
        super(GeneTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)
        ])
        self.avg = nn.AvgPool2d(kernel_size=(1, 200))
        self.output_layer0 = nn.Linear(7, 4)
        self.output_layer1 = nn.Linear(4, 1)

    def forward(self, x):
        # keep original free-variable lookups for is_TPM_format & all_gene_mean_expression
        x = x.view(-1, 7, self.hidden_size)
        x1 = x
        x_copy = x
        if is_TPM_format:
            x1 = torch.log2(x1 + 1)
        gene_ratio = torch.sum(x_copy.reshape(-1)) / all_gene_mean_expression
        x1 = x1 / gene_ratio
        x1 = nn.LayerNorm(self.hidden_size)(x1)
        for layer in self.transformer_layers:
            x1 = layer(x1)
        x1 = self.avg(x1)
        x1 = x1.view(x.size(0), -1)
        x1 = self.output_layer0(x1)
        x1 = nn.ReLU()(x1)
        x1 = self.output_layer1(x1)
        x1 = 4 * nn.Tanh()(x1)
        x1 = torch.sigmoid(x1)
        return x1

# ---------------- Core pipeline as functions ----------------
def _load_pathways(pathways_csv: str):
    """Load pathways CSV and build gene_set exactly like the original code."""
    df = pd.read_csv(pathways_csv)
    df_copy = df.copy()
    df_copy[df_copy.isnull()] = "null_gene"
    gene_set = []
    for i in range(df_copy.shape[1]):
        gene_set += list(df_copy.iloc[:, i])
    # for missing gene logic
    df_t = np.array(df.copy().T)
    df_flat = pd.DataFrame(df_t.reshape(-1)).dropna().values.reshape(-1)
    return df, gene_set, df_flat

def _prepare_inputs(input_csv: str, pathways_csv: str):
    """
    Prepare the input tensor exactly as the original code,
    but avoid fragmented DataFrame by batching new columns.
    """
    pathways_df, gene_set, dropna_flat = _load_pathways(pathways_csv)

    # original reshape helpers
    pathways_copy1 = pathways_df.copy()
    pathways_copy2 = np.array(pathways_copy1.T)
    pathways_copy3 = pd.DataFrame(pathways_copy2.reshape(-1))
    pathways_copy4 = np.array(pathways_copy3.dropna())
    selected_genes_raw = np.array(gene_set)
    selected_genes_raw1 = selected_genes_raw.reshape(-1, 200)
    selected_genes_dropna = pathways_copy4.reshape(-1)

    # read expression and prepare tables
    expr = pd.read_csv(input_csv, index_col=0)  # genes x samples (TPM expected)
    expr_T = expr.T.copy()                     # samples x genes
    expr_T["null_gene"] = 0
    expr_T_copy = expr_T.copy()

    not_available = list(set(selected_genes_dropna) - set(expr.index))
    print(len(not_available))
    # ---- collect new columns first, add once to avoid fragmentation ----
    new_cols = {}
    not_available_set = set(not_available)
    for gene_vector in selected_genes_raw1:
        avail = list(set(gene_vector) - not_available_set)
        to_fill = [g for g in gene_vector if g in not_available_set and g != "null_gene"]
        if not to_fill:
            continue
        imputed = (
            np.power(2, pd.DataFrame(np.log2(expr_T_copy[avail] + 1)).mean(axis=1)) - 1
            if len(avail) > 0 else pd.Series(0.0, index=expr_T.index)
        )
        for g in to_fill:
            new_cols[g] = imputed

    if new_cols:
        add_df = pd.DataFrame(new_cols, index=expr_T.index)
        expr_T = pd.concat([expr_T, add_df], axis=1)

    expr_ordered = expr_T.loc[:, selected_genes_raw]
    x_np = np.array(expr_ordered, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    # sample ids for output index = original columns (samples)
    sample_ids = expr.columns.tolist()
    return x_tensor, sample_ids, gene_set

def _register_pickle_aliases(is_tpm_format_val: bool, agme_val: float):
    """
    Ensure torch.load can find classes & globals at the module where the model was saved.
    Many checkpoints were pickled with __main__.GeneTransformer, so we alias it here.
    Also propagate the current is_TPM_format/all_gene_mean_expression values.
    """
    if '__main__' not in sys.modules:
        sys.modules['__main__'] = types.ModuleType('__main__')
    main_mod = sys.modules['__main__']
    # classes
    setattr(main_mod, 'GeneTransformer', GeneTransformer)
    setattr(main_mod, 'TransformerLayer', TransformerLayer)
    # globals used in forward (propagate current values)
    setattr(main_mod, 'is_TPM_format', is_tpm_format_val)
    setattr(main_mod, 'all_gene_mean_expression', agme_val)

def _load_model(model_pth: str, is_tpm_format_val: bool, agme_val: float):
    """Load full module as in the original code (no restructuring)."""
    # Make sure pickled refs like __main__.GeneTransformer are resolvable
    _register_pickle_aliases(is_tpm_format_val, agme_val)
    # build an instance first (not strictly necessary but safe if unpickling needs class)
    _ = GeneTransformer(hidden_size, num_layers, num_heads, dropout, is_TPM_format)
    net = torch.load(model_pth, map_location="cpu")
    net.eval()
    return net

def predict_file(input_csv: str,
                 pathways_csv: str,
                 model_pth: str,
                 threshold: float = 0.5,
                 *,
                 is_tpm_format: bool = True,
                 all_gene_mean_expression_value: float = 4992.7214) -> pd.DataFrame:
    """
    Predict from file paths. Returns a DataFrame identical in form to the original output,
    with index = sample IDs (input columns), and columns: risk_probability, binary_risk.

    Parameters added:
      - is_tpm_format: whether to apply log2(x+1) inside forward (overrides global).
      - all_gene_mean_expression_value: normalization constant used in forward (overrides global).
    """
    _set_seeds()

    # Override module-level globals so forward() uses the provided values
    global is_TPM_format, all_gene_mean_expression
    is_TPM_format = is_tpm_format
    all_gene_mean_expression = float(all_gene_mean_expression_value)

    x_tensor, sample_ids, _ = _prepare_inputs(input_csv, pathways_csv)

    # placeholders same as original (unused, kept for DataLoader signature)
    test_OS = torch.zeros(x_tensor.shape[0], 1).view(-1)
    test_OS_time = torch.ones(x_tensor.shape[0], 1).view(-1)
    ds = MultitaskDataset(x_tensor, test_OS, test_OS_time)
    loader = torch.utils.data.DataLoader(ds, batch_size=test_OS.shape[0])

    # Load model with aliases that reflect the current two parameters
    net = _load_model(model_pth, is_TPM_format, all_gene_mean_expression)

    with torch.no_grad():
        for batch in loader:
            inputs, _, _ = batch
            inputs = inputs.to(torch.float32)
            outputs = net(inputs)

    probs = np.array(outputs.view(-1))
    df = pd.DataFrame({"risk_probability": probs}, index=sample_ids)
    df["binary_risk"] = np.where(df["risk_probability"] > threshold, "high", "low")
    return df

def predict_df(expr_df: pd.DataFrame,
               pathways_csv: str,
               model_pth: str,
               threshold: float = 0.5,
               *,
               is_tpm_format: bool = True,
               all_gene_mean_expression_value: float = 4992.7214) -> pd.DataFrame:
    """
    Predict from an in-memory expression DataFrame (genes x samples).
    Mirrors predict_file but accepts a DataFrame directly.

    Parameters added:
      - is_tpm_format
      - all_gene_mean_expression_value
    """
    # write temp CSV to reuse exact original pipeline without changing math
    tmp_in = "_tmp_expr_input.csv"
    expr_df.to_csv(tmp_in)
    try:
        return predict_file(tmp_in, pathways_csv, model_pth, threshold,
                            is_tpm_format=is_tpm_format,
                            all_gene_mean_expression_value=all_gene_mean_expression_value)
    finally:
        try:
            os.remove(tmp_in)
        except Exception:
            pass

# -------------- Optional CLI wrapper (can ignore if importing) --------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict (importable minimal wrapper)")
    parser.add_argument("--input", required=True, help="Input CSV (TPM), genes as index; samples as columns")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--model", required=True, help="Trained model .pth path")
    parser.add_argument("--pathways", required=True, help="Deep_learning_input_pathways.csv path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold (default 0.5)")
    parser.add_argument("--is-tpm-format", action="store_true", default=True,
                        help="Apply log2(x+1) inside forward (default True)")
    parser.add_argument("--agme", type=float, default=4992.7214,
                        help="all_gene_mean_expression value (default 4992.7214)")
    args = parser.parse_args()

    out_df = predict_file(args.input, args.pathways, args.model, threshold=args.threshold,
                          is_tpm_format=args.is_tpm_format, all_gene_mean_expression_value=args.agme)
    out_df.to_csv(args.output, index=True)
