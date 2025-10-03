# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:31:25 2023

@author: Wanxiang Shen

"""

"""
Model workflow:
INPUT
[B, 15673] raw data
    ↓
PARSE
cancer_types: [B]
genes: [B, 15672]
    ↓
EMBED (parallel)
├─ PID:    [B, 1, 32]
├─ Cancer: [B, 1, 32]
└─ Genes:  [B, 15672, 32]
    ↓ (inside GeneEmbedding)
    ├─ weight * expression: [B, 15672, 32]
    ├─ + bias (pos encoding): [B, 15672, 32]
    └─ ReLU: [B, 15672, 32]
    ↓
CONCATENATE
[B, 15674, 32] (PID + Cancer + 15672 genes)
    ↓
TRANSFORMER (self-attention)
[B, 15674, 32] → [B, 15674, 32] (contextualized)
    ↓
PROJECTOR
[B, 15674, 32] → [B, 132] → [B, 44]
(genes → gene sets → concepts)
    ↓
DECODER
[B, 44] → [B, 2]
    ↓
OUTPUT
Logits: [B, 2]
"""

import torch
import torch.nn as nn
from ..encoder import TransformerEncoder, MLPEncoder, IdentityEncoder
from ..decoder import ClassDecoder, RegDecoder, ProtoNetDecoder
from ..projector import DisentangledProjector, EntangledProjector


import numpy as np
import random


def fixseed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Compass(nn.Module):

    def __init__(
        self,
        input_dim,
        task_dim,
        task_type,
        num_cancer_types=33,
        embed_dim=32,
        #### projections
        proj_disentangled=True,
        proj_level="cellpathway",
        proj_pid=False,
        proj_cancer_type=True,
        ref_for_task=True,
        encoder="transformer",
        encoder_dropout=0.0,
        task_dense_layer=[24],
        task_batch_norms=True,
        transformer_dim=32,
        transformer_num_layers=1,
        transformer_nhead=2,
        transformer_pos_emb="learnable",
        seed=42,
        **encoder_kwargs
    ):
        """
        input_dim:  number of tokens
        task_dim: supervised learning task dim
        task_type: {'r', 'c'}
        num_cancer_types: int, number cancer types, default 33.
        embed_dim: latent vector dim
        encoder: {'transfomer', 'flowformer', ...}
        task_dense_layer: dense layer of task
        transformer_pos_emb: {None, 'umap', 'pumap'}
        """
        super().__init__()

        self.input_dim = input_dim
        self.task_dim = task_dim
        self.task_type = task_type

        self.proj_disentangled = proj_disentangled
        self.proj_level = proj_level
        self.proj_pid = proj_pid
        self.proj_cancer_type = proj_cancer_type
        self.ref_for_task = ref_for_task

        self.num_cancer_types = num_cancer_types
        self.encoder = encoder
        self.encoder_dropout = encoder_dropout
        self.transformer_dim = transformer_dim
        self.transformer_num_layers = transformer_num_layers
        self.transformer_pos_emb = transformer_pos_emb
        self.transformer_nhead = transformer_nhead
        self.task_batch_norms = task_batch_norms
        self.task_dense_layer = task_dense_layer
        self.encoder_kwargs = encoder_kwargs
        self.seed = seed

        fixseed(seed=self.seed)
        
        if encoder == "identity":
            # Use an identity encoder
            # Test whether gene attention is useful
            self.inputencoder = IdentityEncoder(
            num_cancer_types=num_cancer_types,
            input_dim=input_dim,
            d_model=transformer_dim,
            pos_emb=transformer_pos_emb,
            **encoder_kwargs
        )
        
        else:
            # Gene language model: contextualize each gene using self-attention
            self.inputencoder = TransformerEncoder(
                num_cancer_types=num_cancer_types,
                encoder_type=encoder,
                input_dim=input_dim,
                d_model=transformer_dim,
                num_layers=transformer_num_layers,
                nhead=transformer_nhead,
                dropout=encoder_dropout,
                pos_emb=transformer_pos_emb,
                **encoder_kwargs
            )

        if proj_disentangled:
            # Implement concept bottleneck architecture:
            # Maps gene embeddings to 132 granular concepts (gene sets)
            # Aggregates them into 44 high-level concepts (cell types/pathways)
            
            self.latentprojector = DisentangledProjector(
                input_dim,
                transformer_dim,
                proj_pid=proj_pid,
                proj_cancer_type=proj_cancer_type,
            )

            self.geneset_feature_name = self.latentprojector.geneset_proj_cols
            self.celltype_feature_name = self.latentprojector.cellpathway_proj_cols
            self.ref_gene_ids = self.latentprojector.ref_gene_ids
            self.ref_geneset_ids = self.latentprojector.ref_geneset_ids
            self.ref_celltype_ids = self.latentprojector.ref_celltype_ids

            if proj_level == "cellpathway":
                a = self.celltype_feature_name
                k = self.ref_celltype_ids
            else:
                a = self.geneset_feature_name
                k = self.ref_geneset_ids

            if not self.ref_for_task:
                self.embed_dim = len(a) - len(k)
                self.embed_feature_names = a
                self.proj_feature_names = [a[i] for i in range(len(a)) if i not in k]
            else:
                self.embed_dim = len(a)
                self.embed_feature_names = a
                self.proj_feature_names = a

        else:
            # Baseline/ablation: non-interpretable projection (no gene signatures)
            
            self.latentprojector = EntangledProjector(transformer_dim)
            self.embed_feature_names = range(len(embed_dim))
            self.embed_dim = embed_dim
            self.proj_feature_names = self.embed_feature_names

        model_args = {
            "input_dim": self.input_dim,
            "task_dim": self.task_dim,
            "task_type": self.task_type,
            "proj_level": self.proj_level,
            "proj_pid": self.proj_pid,
            "proj_cancer_type": self.proj_cancer_type,
            "proj_disentangled": self.proj_disentangled,
            "embed_dim": self.embed_dim,
            "num_cancer_types": self.num_cancer_types,
            "encoder": self.encoder,
            "encoder_dropout": self.encoder_dropout,
            "transformer_dim": self.transformer_dim,
            "transformer_nhead": self.transformer_nhead,
            "transformer_num_layers": self.transformer_num_layers,
            "transformer_pos_emb": self.transformer_pos_emb,
            "task_batch_norms": self.task_batch_norms,
            "task_dense_layer": self.task_dense_layer,
            "seed": self.seed,
        }

        model_args.update(encoder_kwargs)
        self.model_args = model_args

        ## regression task
        if task_type == "r":
            self.taskdecoder = RegDecoder(
                input_dim=self.embed_dim,
                dense_layers=task_dense_layer,
                out_dim=task_dim,
                batch_norms=task_batch_norms,
                seed=self.seed,
            )

        ## classification task
        elif task_type == "c":
            self.taskdecoder = ClassDecoder(
                input_dim=self.embed_dim,
                dense_layers=task_dense_layer,
                out_dim=task_dim,
                batch_norms=task_batch_norms,
                seed=self.seed,
            )

        # for softmax classifier
        elif task_type == "f":
            self.taskdecoder = ProtoNetDecoder(
                input_dim=self.embed_dim,
                out_dim=task_dim,
                dense_layers=task_dense_layer,
                batch_norms=task_batch_norms,
                seed=self.seed,
            )

    def forward(self, x):

        # output： B,L+2, (dataset:1, cancer:1, gene),C
        encoding = self.inputencoder(x)
        
        # geneset_level_proj: [B, 132] (granual concepts)
        # cellpathway_level_proj: [B, 44] (high-level concepts)
        geneset_level_proj, cellpathway_level_proj = self.latentprojector(encoding)

        # task_inputs: only input the context-oriented features (for downstream task)
        # Reference refers to houskeeping genes/pathways
        # Embedding: embeddings for contrastive learning
        if self.proj_level == "geneset":
            embedding = geneset_level_proj  # B,L
            emb_ref = embedding[:, self.ref_geneset_ids]

            # Create a mask to exclude reference concepts
            mask = torch.ones(embedding.shape[1], dtype=torch.bool)
            mask[self.ref_geneset_ids] = False
            emb_used = embedding[:, mask]

        elif self.proj_level == "cellpathway":
            embedding = cellpathway_level_proj  # B,L
            emb_ref = embedding[:, self.ref_celltype_ids]

            mask = torch.ones(embedding.shape[1], dtype=torch.bool)
            mask[self.ref_celltype_ids] = False
            emb_used = embedding[:, mask]
            # print(emb_used.shape)

        # Whether to use reference genes/concepts
        if self.ref_for_task:
            y = self.taskdecoder(embedding)
        else:
            y = self.taskdecoder(emb_used)

        gene_encoding = encoding[:, 2:, :]
        gene_ref = gene_encoding[:, self.ref_gene_ids, :]

        # embedding: [B, 44] - high-level concept scores
        # (gene_ref, emb_ref): Tuple of reference features
        # y: [B, 2] - prediction logits
        return (embedding, (gene_ref, emb_ref)), y

