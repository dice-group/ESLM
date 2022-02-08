#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:30:53 2022

@author: asep
"""

import os
import argparse
import torch
from torch import optim
from transformers import BertTokenizer, BertConfig

from config import config
from helpers import Utils
from model import BERTGATES
from dataset import ESBenchmark
from graphs_representation import GraphRepresentation

UTILS = Utils()
LOSS_FUNCTION = config["loss_function"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    if mode == "train":
        log_file_path = os.path.join(os.getcwd(), 'GATES_log.txt')
        with open(log_file_path, 'w', encoding="utf-8") as log_file:
            pass
    for ds_name in config["ds_name"]:
        graph_r = GraphRepresentation(ds_name)
        if mode == "train":
            for topk in config["topk"]:
                dataset = ESBenchmark(ds_name, file_n, topk, is_weighted_adjacency_matrix)
                train_data, valid_data = dataset.get_training_dataset()
                best_epochs = []
                for fold in range(5):
                    print(fold, f"total entities: {len(train_data[fold][0])}", f"topk: top{topk}")
                    model = BERTGATES(bert_config)
                    model.to(DEVICE)
                    if config["regularization"] is True:
                        optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=w_decay)
                    else:
                        optimizer = optim.Adam(model.parameters(), lr=lrate)
                    models_path=os.path.join("models", f"bert_gates_checkpoint-{ds_name}-{topk}-{fold}")
                    models_dir = os.path.join(os.getcwd(), models_path)
                    best_epoch = train(model, optimizer, train_data[fold][0], valid_data[fold][0], dataset, graph_r, topk, fold, models_dir)
                    best_epochs.append(best_epoch)
                with open(log_file_path, 'a', encoding="utf-8") as log_file:
                    line = f'{ds_name}-top{topk} epoch:\t{best_epochs}\n'
                    log_file.write(line)
def train(model, optimizer, train_data, valid_data, dataset, graph, topk, fold, models_dir):
    """Training module"""
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    best_acc=0
    stop_valid_epoch = None
    for epoch in range(config["n_epochs"]):
        model.train()
        train_loss = 0
        train_acc = 0
        for eid in train_data:
            optimizer.zero_grad()
            triples = dataset.get_triples(eid)
            literal = dataset.get_literals(eid)
            adj = graph.build_graph(triples, literal, eid)
            labels = dataset.prepare_labels(eid)
            features = UTILS.convert_to_features(literal, TOKENIZER, MAX_LENGTH, triples, labels)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            target_tensor = UTILS.tensor_from_weight(len(triples), triples, labels)
            output_tensor = model(adj, all_input_ids, all_segment_ids, all_input_mask)
            loss = LOSS_FUNCTION(output_tensor.view(-1), target_tensor.view(-1)).to(DEVICE)
            train_output_tensor = output_tensor.view(1, -1).cpu()
            #train_target_tensor = target_tensor.view(1, -1).cpu()
            #(label_top_scores, label_top) = torch.topk(train_target_tensor, topk)
            (_, output_top) = torch.topk(train_output_tensor, topk)
            triples_dict = dataset.triples_dictionary(eid)
            gold_list_top = dataset.get_gold_summaries(eid, triples_dict)
            acc = UTILS.accuracy(output_top.squeeze(0).numpy().tolist(), gold_list_top)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc
        model.eval()
        valid_acc = 0
        valid_loss = 0
        with torch.no_grad():
            for eid in valid_data:
                triples = dataset.get_triples(eid)
                literal = dataset.get_literals(eid)
                adj = graph.build_graph(triples, literal, eid)
                labels = dataset.prepare_labels(eid)
                features = UTILS.convert_to_features(literal, TOKENIZER, MAX_LENGTH, triples, labels)
                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
                target_tensor = UTILS.tensor_from_weight(len(triples), triples, labels)
                output_tensor = model(adj, all_input_ids, all_segment_ids, all_input_mask)
                loss = LOSS_FUNCTION(output_tensor.view(-1), target_tensor.view(-1)).to(DEVICE)
                train_output_tensor = output_tensor.view(1, -1).cpu()
                #train_target_tensor = target_tensor.view(1, -1).cpu()
                #(label_top_scores, label_top) = torch.topk(train_target_tensor, topk)
                (_, output_top) = torch.topk(train_output_tensor, topk)
                triples_dict = dataset.triples_dictionary(eid)
                gold_list_top = dataset.get_gold_summaries(eid, triples_dict)
                acc = UTILS.accuracy(output_top.squeeze(0).numpy().tolist(), gold_list_top)
                valid_loss += loss.item()
                valid_acc += acc
        train_loss = train_loss/len(train_data)
        train_acc = train_acc/len(train_data)
        valid_loss = valid_loss/len(valid_data)
        valid_acc = train_acc/len(valid_data)
        print(f"epoch: {epoch}, train-loss:{train_loss}, train-acc:{train_acc}, valid-loss:{valid_loss}, valid-acc:{valid_acc}")
        if valid_acc > best_acc:
            print(f"saving best model,  val_accuracy improved from {best_acc} to {valid_acc}")
            best_acc = valid_acc
            torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    'valid_loss': valid_loss,
                    'fold': fold,
                    'acc': best_acc
                    }, os.path.join(models_dir, f"checkpoint_epoch_{epoch}.pt"))
            if os.path.exists(os.path.join(models_dir, f"checkpoint_epoch_{stop_valid_epoch}.pt")):
                os.remove(os.path.join(models_dir, "checkpoint_epoch_{stop_valid_epoch}.pt"))
            stop_valid_epoch = epoch
    return stop_valid_epoch
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='BERT-GATES')
    PARSER.add_argument("--mode", type=str, default="test", help="mode type: train/test/all")
    ARGS = PARSER.parse_args()
    main(ARGS.mode)
    