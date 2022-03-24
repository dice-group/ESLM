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
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BertModel
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from rich.console import Console

from evaluator.map import MAP
from evaluator.fmeasure import FMeasure
from evaluator.ndcg import NDCG
from config import config
from classes.helpers import Utils
from classes.dataset import ESBenchmark

UTILS = Utils()
LOSS_FUNCTION = config["loss_function"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_model='bert-base-uncased'
TOKENIZER = AutoTokenizer.from_pretrained(pretrained_model)
MAX_LENGTH = 39
# define a rich console logger
console=Console(record=True)
    
class BertClassifier(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', nb_class=1):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = nn.Linear(self.feat_dim, nb_class)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
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
def main(mode):
    """Main module"""
    file_n = config["file_n"]
    is_weighted_adjacency_matrix = config["weighted_adjacency_matrix"]
    if mode == "train":
        # logging
        console.log(f"""Checking logging directory ...""")
        log_file_path = os.path.join(os.getcwd(), 'logs/Bert_log.txt')
        if os.path.exists(os.path.join(os.getcwd(), "logs")) is not True:
            console.log(f"""Creating logging directory ...""")
            os.mkdir(os.path.join(os.getcwd(), "logs"))
        with open(log_file_path, 'w', encoding="utf-8") as log_file:
            pass
    for ds_name in config["ds_name"]:
        if mode == "test":
            for topk in config["topk"]:
                filename = 'logs/Bert_log.txt'
                use_epoch = UTILS.read_epochs_from_log(ds_name, topk, filename)
                dataset = ESBenchmark(ds_name, file_n, topk, is_weighted_adjacency_matrix)
                test_data = dataset.get_testing_dataset()
                fmeasure_scores = []
                ndcg_scores = []
                map_scores = []
                for fold in range(5):
                    print("")
                    print(f"fold: {fold+1}, total entities: {len(test_data[fold][0])}", f"topk: top{topk}")
                    models_path = os.path.join("models", f"bert_checkpoint-{ds_name}-{topk}-{fold}")
                    model = BertClassifier()
                    checkpoint = torch.load(os.path.join(models_path, f"checkpoint_epoch_{use_epoch[fold]}.pt"))
                    model.bert_model.load_state_dict(checkpoint["bert_model"])
                    model.classifier.load_state_dict(checkpoint["classifier"])
                    model.to(DEVICE)
                    fmeasure_score, ndcg_score, map_score = generated_entity_summaries(model, test_data[fold][0], dataset, topk)
                    fmeasure_scores.append(fmeasure_score)
                    ndcg_scores.append(ndcg_score)
                    map_scores.append(map_score)
                print(f"{dataset.ds_name}@top{topk}: F-Measure={np.average(fmeasure_scores)}, NDCG={np.average(ndcg_scores)}, MAP={np.average(map_scores)}")

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
            #all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            target_tensor = UTILS.tensor_from_weight(len(triples), triples, labels)
            output_tensor = model(all_input_ids, all_input_mask)
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
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='BERT-GATES')
    PARSER.add_argument("--mode", type=str, default="test", help="mode type: train/test/all")
    ARGS = PARSER.parse_args()
    main(ARGS.mode)
    