#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:25:40 2022

@author: asep
"""
import os
import argparse
import time
import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as f
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from rich.console import Console
from distutils.util import strtobool

from evaluator.map import MAP
from evaluator.fmeasure import FMeasure
from evaluator.ndcg import NDCG
from config import config
from classes.helpers import Utils
from classes.dataset import ESBenchmark

UTILS = Utils()
LOSS_FUNCTION = config["loss_function"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_model='ernie-2.0-en'
TOKENIZER = AutoTokenizer.from_pretrained(pretrained_model)
MAX_LENGTH = 42
# define a rich console logger
console=Console(record=True)
    
class ErnieClassifier(nn.Module):
    def __init__(self, pretrained_model='nghuyong/ernie-2.0-en', nb_class=1):
        super(ErnieClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = nn.Linear(self.feat_dim, nb_class)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)#self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(outputs.pooler_output)
        cls_logit = self.softmax(cls_logit)
        return cls_logit    
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
def main(mode, best_epoch):
    """Main module"""
    file_n = config["file_n"]
    is_weighted_adjacency_matrix = config["weighted_adjacency_matrix"]
    if mode == "train":
        # logging
        console.log(f"""Checking logging directory ...""")
        log_file_path = os.path.join(os.getcwd(), 'logs/Ernie_log.txt')
        if os.path.exists(os.path.join(os.getcwd(), "logs")) is not True:
            console.log(f"""Creating logging directory ...""")
            os.mkdir(os.path.join(os.getcwd(), "logs"))
        with open(log_file_path, 'w', encoding="utf-8") as log_file:
            pass
    for ds_name in config["ds_name"]:
        if mode == "train":
            for topk in config["topk"]:
                dataset = ESBenchmark(ds_name, file_n, topk, is_weighted_adjacency_matrix)
                train_data, valid_data = dataset.get_training_dataset()
                best_epochs = []
                for fold in range(5):
                    fold = fold
                    print("")
                    print(f"Fold: {fold+1}, total entities: {len(train_data[fold][0])}", f"topk: top{topk}")
                    model = ErnieClassifier()
                    model.to(DEVICE)
                    param_optimizer = list(model.named_parameters())
                    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                    optimizer_grouped_parameters = [
                        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                        ]
                    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
                    models_path = os.path.join("models", f"ernie_checkpoint-{ds_name}-{topk}-{fold}")
                    models_dir = os.path.join(os.getcwd(), models_path)
                    best_epoch = train(model, optimizer, train_data[fold][0], valid_data[fold][0], dataset, topk, fold, models_dir)
                    best_epochs.append(best_epoch)
                with open(log_file_path, 'a', encoding="utf-8") as log_file:
                    line = f'{ds_name}-top{topk} epoch:\t{best_epochs}\n'
                    log_file.write(line)
        elif mode == "test":
            for topk in config["topk"]:
                #filename = 'logs/Bert_log.txt'
                #use_epoch = UTILS.read_epochs_from_log(ds_name, topk, filename)
                dataset = ESBenchmark(ds_name, file_n, topk, is_weighted_adjacency_matrix)
                test_data = dataset.get_testing_dataset()
                fmeasure_scores = []
                ndcg_scores = []
                map_scores = []
                for fold in range(5):
                    print("")
                    print(f"fold: {fold+1}, total entities: {len(test_data[fold][0])}", f"topk: top{topk}")
                    models_path = os.path.join("models", f"ernie_checkpoint-{ds_name}-{topk}-{fold}")
                    model = ErnieClassifier()
                    if bool(strtobool(best_epoch)) is True:
                        checkpoint = torch.load(os.path.join(models_path, f"checkpoint_best_{fold}.pt"))
                    else:
                        checkpoint = torch.load(os.path.join(models_path, f"checkpoint_latest_{fold}.pt"))
                    model.bert_model.load_state_dict(checkpoint["bert_model"])
                    model.classifier.load_state_dict(checkpoint["classifier"])
                    model.to(DEVICE)
                    fmeasure_score, ndcg_score, map_score = generated_entity_summaries(model, test_data[fold][0], dataset, topk)
                    fmeasure_scores.append(fmeasure_score)
                    ndcg_scores.append(ndcg_score)
                    map_scores.append(map_score)
                print(f"{dataset.ds_name}@top{topk}: F-Measure={np.average(fmeasure_scores)}, NDCG={np.average(ndcg_scores)}, MAP={np.average(map_scores)}")
def train(model, optimizer, train_data, valid_data, dataset, topk, fold, models_dir):
    """Training module"""
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    best_acc = 0
    stop_valid_epoch = None
    total_steps = len(train_data) * config["n_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    for epoch in range(config["n_epochs"]):
        model.train()
        train_loss = 0
        train_acc = 0
        print("")
        print(f'======== Epoch {epoch+1} / {config["n_epochs"]} ========')
        print('Training...')
        t_start = time.time()
        for eid in tqdm(train_data):
            triples = dataset.get_triples(eid)
            literal = dataset.get_literals(eid)
            labels = dataset.prepare_labels(eid)
            features = UTILS.convert_to_features_with_subject(literal, TOKENIZER, MAX_LENGTH, triples, labels)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            target_tensor = UTILS.tensor_from_weight(len(triples), triples, labels)
            output_tensor = model(all_input_ids, all_input_mask, all_segment_ids)
            #print(output_tensor)
            #print(output_tensor.shape)
            loss = LOSS_FUNCTION(output_tensor.view(-1), target_tensor.view(-1)).to(DEVICE)
            train_output_tensor = output_tensor.view(1, -1).cpu()
            (_, output_top) = torch.topk(train_output_tensor, topk)
            triples_dict = dataset.triples_dictionary(eid)
            gold_list_top = dataset.get_gold_summaries(eid, triples_dict)
            acc = UTILS.accuracy(output_top.squeeze(0).numpy().tolist(), gold_list_top)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            train_acc += acc
        training_time = format_time(time.time() - t_start)
        print("  Training epcoh took: {:}".format(training_time))
        valid_acc = 0
        valid_loss = 0
        print("")
        print("Running Validation...")
        t_start = time.time()
        model.eval()
        with torch.no_grad():
            for eid in tqdm(valid_data):
                triples = dataset.get_triples(eid)
                literal = dataset.get_literals(eid)
                labels = dataset.prepare_labels(eid)
                features = UTILS.convert_to_features_with_subject(literal, TOKENIZER, MAX_LENGTH, triples, labels)
                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
                target_tensor = UTILS.tensor_from_weight(len(triples), triples, labels)
                #output_tensor = model(all_input_ids, all_segment_ids, all_input_mask)
                output_tensor = model(all_input_ids, all_input_mask, all_segment_ids)
                loss = LOSS_FUNCTION(output_tensor.view(-1), target_tensor.view(-1)).to(DEVICE)
                valid_output_tensor = output_tensor.view(1, -1).cpu()
                (_, output_top) = torch.topk(valid_output_tensor, topk)
                triples_dict = dataset.triples_dictionary(eid)
                gold_list_top = dataset.get_gold_summaries(eid, triples_dict)
                acc = UTILS.accuracy(output_top.squeeze(0).numpy().tolist(), gold_list_top)
                valid_loss += loss.item()
                valid_acc += acc
        validation_time = format_time(time.time() - t_start)
        print("  Validation took: {:}".format(validation_time))
        train_loss = train_loss/len(train_data)
        train_acc = train_acc/len(train_data)
        valid_loss = valid_loss/len(valid_data)
        valid_acc = valid_acc/len(valid_data)
        print("")
        print(f"train-loss:{train_loss}, train-acc:{train_acc}, valid-loss:{valid_loss}, valid-acc:{valid_acc}")
        if valid_acc > best_acc:
            print(f"saving best model,  val_accuracy improved from {best_acc} to {valid_acc}")
            best_acc = valid_acc
            torch.save({
                "epoch": epoch,
                "bert_model": model.bert_model.state_dict(),
                "classifier": model.classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                'valid_loss': valid_loss,
                'fold': fold,
                'acc': best_acc,
                'training_time': training_time,
                'validation_time': validation_time
                }, os.path.join(models_dir, f"checkpoint_best_{fold}.pt"))
            stop_valid_epoch = epoch
            
        torch.save({
                "epoch": epoch,
                "bert_model": model.bert_model.state_dict(),
                "classifier": model.classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                'valid_loss': valid_loss,
                'fold': fold,
                'acc': best_acc,
                'training_time': training_time,
                'validation_time': validation_time
                }, os.path.join(models_dir, f"checkpoint_latest_{fold}.pt"))
    return stop_valid_epoch
def generated_entity_summaries(model, test_data, dataset, topk):
    """"Generated entity summaries"""
    model.eval()
    ndcg_eval = NDCG()
    fmeasure_eval = FMeasure()
    map_eval = MAP()
    ndcg_scores = []
    fmeasure_scores = []
    map_scores = []
    with torch.no_grad():
        for eid in test_data:
            triples = dataset.get_triples(eid)
            literal = dataset.get_literals(eid)
            labels = dataset.prepare_labels(eid)
            features = UTILS.convert_to_features_with_subject(literal, TOKENIZER, MAX_LENGTH, triples, labels)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            target_tensor = UTILS.tensor_from_weight(len(triples), triples, labels)
            output_tensor = model(all_input_ids, all_input_mask, all_segment_ids)
            
            #applu similarity 
            #output_tensor = model.bert_model(all_input_ids, all_input_mask)
            #console.log(f"""Calculting the similarity between triples ...""")
            #cls_distance = max_cosine_distance(output_tensor[0])
            #console.log(cls_distance)
            #output_tensor = cls_distance[0]
            output_tensor = output_tensor.view(1, -1).cpu()
            target_tensor = target_tensor.view(1, -1).cpu()
            #(label_top_scores, label_top) = torch.topk(target_tensor, topk)
            _, output_top = torch.topk(output_tensor, topk)
            _, output_rank = torch.topk(output_tensor, len(test_data[eid]))
            triples_dict = dataset.triples_dictionary(eid)
            gold_list_top = dataset.get_gold_summaries(eid, triples_dict)
            top_list_output_top = output_top.squeeze(0).numpy().tolist()
            all_list_output_top = output_rank.squeeze(0).numpy().tolist()
            ndcg_score = ndcg_eval.get_score(gold_list_top, all_list_output_top)
            f_score = fmeasure_eval.get_score(top_list_output_top, gold_list_top)
            map_score = map_eval.get_map(all_list_output_top, gold_list_top)
            ndcg_scores.append(ndcg_score)
            fmeasure_scores.append(f_score)
            map_scores.append(map_score)
            directory = f"outputs/{dataset.get_ds_name}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = f"outputs/{dataset.get_ds_name}/{eid}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            top_or_rank = "top"
            rank_list = output_top.squeeze(0).numpy().tolist()
            writer(dataset.get_db_path, directory, eid, top_or_rank, topk, rank_list)
            top_or_rank = "rank_top"
            rank_list = output_rank.squeeze(0).numpy().tolist()
            writer(dataset.get_db_path, directory, eid, top_or_rank, topk, rank_list)
    return np.average(fmeasure_scores), np.average(ndcg_scores), np.average(map_scores)
def writer(db_dir, directory, eid, top_or_rank, topk, rank_list):
    "Write triples to file"
    with open(os.path.join(db_dir, f"{eid}", f"{eid}_desc.nt"), encoding="utf8") as fin:
        with open(os.path.join(directory, f"{eid}_{top_or_rank}{topk}.nt"), "w", encoding="utf8") as fout:
            triples = [triple for _, triple in enumerate(fin)]
            for rank in rank_list:
                fout.write(triples[rank])
def cls_cosine_distance(embeds):
    CLSs = embeds[:, 0, :]
    # normalize the CLS token embeddings
    normalized = f.normalize(CLSs, p=2, dim=1)
    # calculate the cosine similarity
    cls_dist = normalized.matmul(normalized.T)
    cls_dist = cls_dist.new_ones(cls_dist.shape) - cls_dist
    cls_dist = cls_dist.numpy()
    return cls_dist
def mean_cosine_distance(embeds):
    MEANS = embeds.mean(dim=1)
        # normalize the MEANS token embeddings
    normalized = f.normalize(MEANS, p=2, dim=1)
    # calculate the cosine similarity
    mean_dist = normalized.matmul(normalized.T)
    mean_dist = mean_dist.new_ones(mean_dist.shape) - mean_dist
    return mean_dist
def max_cosine_distance(embeds):
    MAXS, _ = embeds.max(dim=1)
    # normalize the MEANS token embeddings
    normalized = f.normalize(MAXS, p=2, dim=1)
    # calculate the cosine similarity
    max_dist = normalized.matmul(normalized.T)
    max_dist = max_dist.new_ones(max_dist.shape) - max_dist
    return max_dist
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='BERT-GATES')
    PARSER.add_argument("--mode", type=str, default="test", help="mode type: train/test/all")
    PARSER.add_argument("--best_epoch", type=str, default="True", help="")
    ARGS = PARSER.parse_args()
    main(ARGS.mode, ARGS.best_epoch)
    