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
from transformers import get_linear_schedule_with_warmup
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
pretrained_model='nghuyong/ernie-2.0-base-en'
TOKENIZER = AutoTokenizer.from_pretrained(pretrained_model)
# define a rich console logger
console=Console(record=True)
    
class ErnieClassifier(nn.Module):
    def __init__(self, pretrained_model='nghuyong/ernie-2.0-base-en', nb_class=1):
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
    for ds_name in config["ds_name"]:
        if ds_name == "dbpedia":
            MAX_LENGTH = 46
        elif ds_name == "faces":
            MAX_LENGTH = 46
        else:
            MAX_LENGTH = 34
        if mode == "train":
            for topk in config["topk"]:
                dataset = ESBenchmark(ds_name, file_n, topk, is_weighted_adjacency_matrix)
                train_data, valid_data = dataset.get_training_dataset()
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
                    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
                    models_path = os.path.join("models", f"ernie_checkpoint-{ds_name}-{topk}-{fold}")
                    models_dir = os.path.join(os.getcwd(), models_path)
                    train(model, optimizer, train_data[fold][0], valid_data[fold][0], dataset, topk, fold, models_dir, MAX_LENGTH)
        elif mode == "test":
            for topk in config["topk"]:
                dataset = ESBenchmark(ds_name, file_n, topk, is_weighted_adjacency_matrix)
                test_data = dataset.get_testing_dataset()
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
                    generated_entity_summaries(model, test_data[fold][0], dataset, topk, MAX_LENGTH)
                evaluation(dataset, topk)
def train(model, optimizer, train_data, valid_data, dataset, topk, fold, models_dir, max_length):
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
            features = UTILS.convert_to_features_with_subject(literal, TOKENIZER, max_length, triples, labels)
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
                features = UTILS.convert_to_features_with_subject(literal, TOKENIZER, max_length, triples, labels)
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
        if valid_acc >= best_acc:
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
def generated_entity_summaries(model, test_data, dataset, topk, max_length):
    """"Generated entity summaries"""
    model.eval()
    with torch.no_grad():
        for eid in test_data:
            triples = dataset.get_triples(eid)
            literal = dataset.get_literals(eid)
            labels = dataset.prepare_labels(eid)
            features = UTILS.convert_to_features_with_subject(literal, TOKENIZER, max_length, triples, labels)
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
def writer(db_dir, directory, eid, top_or_rank, topk, rank_list):
    "Write triples to file"
    with open(os.path.join(db_dir, f"{eid}", f"{eid}_desc.nt"), encoding="utf8") as fin:
        with open(os.path.join(directory, f"{eid}_{top_or_rank}{topk}.nt"), "w", encoding="utf8") as fout:
            triples = [triple for _, triple in enumerate(fin)]
            for rank in rank_list:
                fout.write(triples[rank])
def get_rank_triples(db_path, num, top_n, triples_dict):
  triples=[]
  encoded_triples = []
  filename = os.path.join(db_path, "{}".format(num), "{}_rank.nt".format(num))
  if os.path.exists(os.path.join(db_path, "{}".format(num), "{}_rank_top{}.nt".format(num, top_n))):
      filename = os.path.join(db_path, "{}".format(num), "{}_rank_top{}.nt".format(num, top_n))
  with open(filename, encoding="utf8") as reader:   
    for i, triple in enumerate(reader):
        triple = triple.replace("\n", "").strip()
        triples.append(triple)
        
        encoded_triple = triples_dict[triple]
        encoded_triples.append(encoded_triple)
  return triples, encoded_triples

def get_topk_triples(db_path, num, top_n, triples_dict):
  triples=[]
  encoded_triples = []
  
  with open(os.path.join(db_path, "{}".format(num), "{}_top{}.nt".format(num, top_n)), encoding="utf8") as reader:   
    for i, triple in enumerate(reader):
        triple = triple.replace("\n", "").strip()
        triples.append(triple)
        
        encoded_triple = triples_dict[triple]
        encoded_triples.append(encoded_triple)
  return triples, encoded_triples


def get_all_data(db_path, num, top_n, file_n):
  import glob
  triples_dict = {}
  triple_tuples = []
  ### Retrieve all triples of an entity based on eid
  with open(os.path.join(db_path, "{}".format(num), "{}_desc.nt".format(num)), encoding="utf8") as reader:   
    for i, triple in enumerate(reader):
      if len(triple)==1:
        continue  
      triple_tuple = triple.replace("\n", "").strip()#parserline(triple)
      triple_tuples.append(triple_tuple)
      if triple_tuple not in triples_dict:
        triples_dict[triple_tuple] = len(triples_dict)
  gold_list = []
  ds_name = db_path.split("/")[-1].split("_")[0]
  
  ### Get file_n/ n files of ground truth summaries for faces dataset
  if ds_name=="faces":
      gold_files = glob.glob(os.path.join(db_path, "{}".format(num), "{}_gold_top{}_*".format(num, top_n).format(num)))
      #print(len(gold_files))
      if len(gold_files) != file_n:
          file_n = len(gold_files)
  
  ### Retrieve ground truth summaries of an entity based on eid and total of file_n  
  for i in range(file_n):
    with open(os.path.join(db_path, 
            "{}".format(num), 
            "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)),
            encoding="utf8") as reader:
      #print(path.join(db_path, "{}".format(num), "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)))
      n_list = []
      for i, triple in enumerate(reader):
        if len(triple)==1:
            continue
        triple_tuple = triple.replace("\n", "").strip()#parserline(triple)
        gold_id = triples_dict[triple_tuple]
        n_list.append(gold_id)
      gold_list.append(n_list)
        
  return gold_list, triples_dict, triple_tuples

def evaluation(dataset, k):
    ndcg_class = NDCG()
    fmeasure = FMeasure()
    m = MAP()
    
    if dataset.ds_name == "dbpedia":
        IN_SUMM = os.path.join(os.getcwd(), 'outputs/dbpedia')
        start = [0, 140]
        end   = [100, 165]
    elif dataset.ds_name == "lmdb":
        IN_SUMM = os.path.join(os.getcwd(), 'outputs/lmdb')
        start = [100, 165]
        end   = [140, 175]
    elif dataset.ds_name == "faces":
        IN_SUMM = os.path.join(os.getcwd(), 'outputs/faces')
        start = [0, 25]
        end   = [25, 50]
            
    all_ndcg_scores = []
    all_fscore = []
    all_map_scores = []
    total_ndcg=0
    total_fscore=0
    total_map_score=0
    for i in range(start[0], end[0]):
        t = i+1
        gold_list_top, triples_dict, triple_tuples = get_all_data(dataset.db_path, t, k, dataset.file_n)
        rank_triples, encoded_rank_triples = get_rank_triples(IN_SUMM, t, k, triples_dict)
        topk_triples, encoded_topk_triples = get_topk_triples(IN_SUMM, t, k, triples_dict)
        #print("############### Top-K Triples ################", t)
        #print("######################")
        #print(triples_dict)
        #print("total of gold summaries", len(gold_list_top))
        #print("topk", encoded_topk_triples)
        #ndcg_score = getNDCG(rel)
        ndcg_score = ndcg_class.get_score(gold_list_top, encoded_rank_triples)
        f_score = fmeasure.get_score(encoded_topk_triples, gold_list_top)
        map_score = m.get_map(encoded_rank_triples, gold_list_top)
        #print(ndcg_score)
        #print("*************************")
        total_ndcg += ndcg_score
        all_ndcg_scores.append(ndcg_score)
        
        total_fscore += f_score
        all_fscore.append(f_score)
        
        all_map_scores.append(map_score)
        
    for i in range(start[1], end[1]):
        t = i+1
        gold_list_top, triples_dict, triple_tuples = get_all_data(dataset.db_path, t, k, dataset.file_n)
        rank_triples, encoded_rank_triples = get_rank_triples(IN_SUMM, t, k, triples_dict)
        topk_triples, encoded_topk_triples = get_topk_triples(IN_SUMM, t, k, triples_dict)
        #print("############### Top-K Triples ################", t)
        #print("######################")
        #print(triples_dict)
        #print("total of gold summaries", len(gold_list_top))
        #print("topk", encoded_topk_triples)
        #ndcg_score = getNDCG(rel)
        ndcg_score = ndcg_class.get_score(gold_list_top, encoded_rank_triples)
        f_score = fmeasure.get_score(encoded_topk_triples, gold_list_top)
        map_score = m.get_map(encoded_rank_triples, gold_list_top)
        #print(ndcg_score)
        #print("*************************")
        total_ndcg += ndcg_score
        all_ndcg_scores.append(ndcg_score)
        
        total_fscore += f_score
        all_fscore.append(f_score)
        
        all_map_scores.append(map_score)
        
    print("{}@top{}: F-Measure={}, NDCG={}, MAP={}".format(dataset, k, np.average(all_fscore), np.average(all_ndcg_scores), np.average(all_map_scores)))
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
    