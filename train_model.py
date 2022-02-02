#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 00:53:56 2022

@author: asep
"""
import os
import time
from torch import optim
from helpers import Utils
from config import config
from model import GATES
from dataset import ESBenchmark
from graphs_representation import GraphRepresentation

dataset = ESBenchmark()
utils = Utils()

def train_iter(ds_name, entity_dict, pred_dict, entity2ix_size, pred2ix_size, text_embed):
    
    # defines variables
    k_fold = []
    times = []
    valid_epoch_list = []
    losses = {'Training set':[], 'Validation set': []}
    
    # monitor time and memory
    start = time.time()
    mem = utils.mem()
    
    # load dataset
    train_data, train_valid = dataset.get_training_dataset(ds_name)

    print("Start time: {}".format(utils.asHours(start)))
    print("Current memory ussage: {}".format(mem))
    
    # check models directory to store GATES model
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # implements k-fold cross validation
    for i in range(5):
        k_fold.append(i)
        
        #defines the model
        gates = GATES(pred2ix_size, entity2ix_size)
        gates.to(config["device"])
        print(gates)
        
        if config["regularization"]:
            optimizer = optim.Adam(gates.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        else:    
            optimizer = optim.Adam(gates.parameters(), lr=config["learning_rate"])
        print(optimizer)
        topk=5
        directory = os.path.join(os.getcwd(), os.path.join("models", "gates_checkpoint-{}-{}-{}".format(ds_name, topk, i)))
        print("Training GATES model on Fold {} on top {} of {} dataset".format(i+1, topk, ds_name))
        train(gates, ds_name, optimizer, text_embed, entity_dict, pred_dict, train_data[i])
        
def train(model, ds_name, optimizer, text_embed, entity_dict, pred_dict, train_data):
    graph_representation = GraphRepresentation()
    ar_epochs = []
    for epoch in range(1):
        model.train()
        ar_epochs.append(epoch)
        train_loss = 0
        train_acc =[]
        for data in train_data:
            for eid in data.keys():
                optimizer.zero_grad()
                print("eid:{}".format(eid))
                facts = data[eid]
                literals = dataset.get_literals(ds_name, eid)
                adj = graph_representation.build_graph(facts, literals, config["weighted_score"], ds_name, eid)
                output_tensor = model(literals, adj)
                
    