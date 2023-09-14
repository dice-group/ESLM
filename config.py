#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 21:56:23 2022

@author: asep
"""
import torch

config = {
        "file_n": 6,
        "topk": [5, 10],
        "ds_name" : ['lmdb', 'faces'],
        "weighted_score"  : "tf-idf",
        "loss_function" : torch.nn.BCELoss(),
        "weight_decay": 1e-5,
        "learning_rate":2e-5,
        "dropout": 0.0,
        "regularization": False,
        "nheads": 2,
        "hidden_layer": 2,
        "entity_embedding_dim": 300,
        "predicate_embedding_dim": 300,
        "n_epochs": 5,
        "weighted_adjacency_matrix": True,
        "device": "cpu", #torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "text_embed_calc_method": "AVG"
}
