#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 21:56:23 2022

@author: asep
"""
import torch

config = {
        "file_n": 6,
        "top_k": [5, 10],
        "ds_name" : ['dbpedia'],
        "weighted_score"  : "pagerank",
        "loss_function" : torch.nn.BCELoss(),
        "weight_decay": 1e-5,
        "learning_rate":0.01,
        "dropout": 0.0,
        "regularization": False,
        "nheads": 2, 
        "hidden_layer": 1,
        "entity_embedding_dim": 300,
        "predicate_embedding_dim": 300,
        "n_epochs": 50,
        "weighted_adjacency_matrix": True,
        "device": torch.device("cpu"),
        "text_embed_calc_method": "AVG"
        
}