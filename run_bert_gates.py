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
from model import BERT_GATES
from dataset import ESBenchmark
from graphs_representation import GraphRepresentation

def main(mode):
    if config["regularization"]==True:
        weight_decay=config["weight_decay"]
    else:
        weight_decay=0
    utils = Utils()
    graph = GraphRepresentation()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    bert_config = BertConfig.from_pretrained("bert-base-cased", num_labels=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_function = config["loss_function"]
    max_seq_length = 16
    for ds_name in config["ds_name"]:
        dataset = ESBenchmark()
        if mode =="train":
            train_data, valid_data = dataset.get_training_dataset(ds_name)
            for topk in config["topk"]:
                for fold in range(5):
                    print(fold, f"total entities: {train_data[fold][0]}", f"topk: top{topk}")
                    model = BERT_GATES(bert_config, config)
                    model.to(device)
                    if config["regularization"]:
                        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=weight_decay)
                    else:    
                        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
                    for epoch in range(config["n_epochs"]):
                        model.train()
                        train_loss = 0
                        for eid in train_data[fold][0].keys():
                            optimizer.zero_grad()
                            triples = dataset.get_triples(ds_name, eid)
                            literals = dataset.get_literals(ds_name, eid)
                            adj = graph.build_graph(triples, literals, config["weighted_score"], ds_name, eid)
                            labels = dataset.prepare_labels(ds_name, eid, topk, config["file_n"])
                            features = utils.convert_to_features(literals, tokenizer, max_seq_length, triples, labels)
                            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
                            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
                            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
                            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
                            logits = model(adj, all_input_ids, all_segment_ids, all_input_mask)
                            loss = loss_function(logits.view(-1), all_label_ids.view(-1)).to(device)
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()
                        train_loss = train_loss/len(train_data[fold][0])
                        print("epoch: {}, loss:{}".format(epoch, train_loss))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GATES: Graph Attention Networks for Entity Summarization')
    parser.add_argument("--mode", type=str, default="test", help="use which mode type: train/test/all")
    args = parser.parse_args()
    main(args.mode)
    