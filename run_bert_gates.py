#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:30:53 2022

@author: asep
"""

import argparse
import torch
from tqdm import tqdm
from torch import optim
from transformers import BertTokenizer, BertConfig

from config import config
from helpers import Utils
from model import BERTGATES
from dataset import ESBenchmark
from graphs_representation import GraphRepresentation

UTILS = Utils()
LOSS_FUNCTION = config["loss_function"]
DEVICE = config["device"]
TOKENIZER = BertTokenizer.from_pretrained("bert-base-cased")
MAX_LENGTH = 16
def main(mode):
    """Main module"""
    if config["regularization"] is True:
        w_decay = config["weight_decay"]
    else:
        w_decay = 0
    bert_config = BertConfig.from_pretrained("bert-base-cased", num_labels=1)
    file_n = config["file_n"]
    is_weighted_adjacency_matrix = config["weighted_adjacency_matrix"]
    lrate = config["learning_rate"]
    for ds_name in config["ds_name"]:
        graph_r = GraphRepresentation(ds_name)
        if mode =="train":
            for topk in config["topk"]:
                dataset = ESBenchmark(ds_name, file_n, topk, is_weighted_adjacency_matrix)
                train_data, _ = dataset.get_training_dataset()
                for fold in range(5):
                    print(fold, f"total entities: {train_data[fold][0]}", f"topk: top{topk}")
                    model = BERTGATES(bert_config, config)
                    model.to(DEVICE)
                    if config["regularization"] is True:
                        optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=w_decay)
                    else:
                        optimizer = optim.Adam(model.parameters(), lr=lrate)
                    train(model, optimizer, train_data[fold][0], dataset, graph_r)
def train(model, optimizer, train_data, dataset, graph):
    """Training module"""
    for epoch in range(config["n_epochs"]):
        model.train()
        train_loss = 0
        for eid in train_data.keys():
            optimizer.zero_grad()
            triples = dataset.get_triples(eid)
            literal = dataset.get_literals(eid)
            adj = graph.build_graph(triples, literal, eid)
            labels = dataset.prepare_labels(eid)
            features_label = UTILS.tensor_from_weight(len(triples), triples, labels)
            features = UTILS.convert_to_features(literal, TOKENIZER, MAX_LENGTH, features_label)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
            logits = model(adj, all_input_ids, all_segment_ids, all_input_mask)
            loss = LOSS_FUNCTION(logits.view(-1), all_label_ids.view(-1)).to(DEVICE)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/len(train_data)
        print(f"epoch: {epoch}, loss:{train_loss}")
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='BERT-GATES')
    PARSER.add_argument("--mode", type=str, default="test", help="mode type: train/test/all")
    ARGS = PARSER.parse_args()
    main(ARGS.mode)
    