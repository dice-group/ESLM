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
import torch.nn.functional as F
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BertModel
from tqdm import tqdm
import scipy.sparse as sp
from distutils.util import strtobool

from evaluator.map import MAP
from evaluator.fmeasure import FMeasure
from evaluator.ndcg import NDCG
from config import config
from classes.helpers import Utils
from classes.dataset import ESBenchmark
from classes.graphs_representation import GraphRepresentation
from transformers import AutoModel, AutoTokenizer

UTILS = Utils()
LOSS_FUNCTION = config["loss_function"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOKENIZER = BertTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
DROPOUT = config["dropout"]
IS_WEIGHTED_ADJ = config["weighted_adjacency_matrix"]
IN_EDGE_FEAT = 1
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_feats, out_feats, alpha, concat=True):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.alpha = alpha
        self.concat = concat
        self.weight = nn.Parameter(torch.empty(size=(in_feats, out_feats)))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if IS_WEIGHTED_ADJ is False:
            self.att = nn.Parameter(torch.empty(size=(2*out_feats, 1)))
        else:
            self.att = nn.Parameter(torch.empty(size=(3*out_feats, 1)))
        nn.init.xavier_uniform_(self.att.data, gain=1.414)
        self.w_edge = nn.Parameter(torch.empty(size=(IN_EDGE_FEAT, out_feats)))
        nn.init.xavier_uniform_(self.w_edge.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, h_nodes, edge, adj):
        """Forward"""
        w_h_nodes = torch.mm(h_nodes, self.weight)
        if IS_WEIGHTED_ADJ is True:
            weight_edges = torch.mm(edge, self.w_edge)
            a_input = self._prepare_att_mechanism_input_with_edge_features(w_h_nodes, weight_edges)
        else:
            a_input = self._prepare_att_mechanism_input(w_h_nodes)
        e_var = self.leakyrelu(torch.matmul(a_input, self.att).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e_var)
        attention = torch.where(adj > 0, e_var, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, DROPOUT, training=self.training)
        h_nodes_prime = torch.matmul(attention, w_h_nodes)
        if self.concat:
            return F.elu(h_nodes_prime)
        return h_nodes_prime
    def _prepare_att_mechanism_input(self, w_h_nodes):
        """prepare attentional mechanism input"""
        nodes = w_h_nodes.size()[0]
        wh_repeated_in_chunks = w_h_nodes.repeat_interleave(nodes, dim=0)
        wh_repeated_alt = w_h_nodes.repeat(nodes, 1)
        all_combinations_matrix = torch.cat([wh_repeated_in_chunks, wh_repeated_alt], dim=1)
        return all_combinations_matrix.view(nodes, nodes, 2 * self.out_feats)
    def _prepare_att_mechanism_input_with_edge_features(self, w_h_nodes, weight_edges):
        nodes = w_h_nodes.size()[0] # number of nodes
        wh_repeated = w_h_nodes.repeat_interleave(nodes, dim=0)
        wh_repeated_alt = w_h_nodes.repeat(nodes, 1)
        weight_edges_repeated = weight_edges.repeat_interleave(nodes, dim=0)
        all_matrix = torch.cat([wh_repeated, wh_repeated_alt, weight_edges_repeated], dim=1)
        return all_matrix.view(nodes, nodes, 3 * self.out_feats)
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_feats) + '->' + str(self.out_feats) + ')'

class GAT(nn.Module):
    """GAT model"""
    def __init__(self, nfeat, nhid, nclass, alpha):
        """Dense version of GAT."""
        super().__init__()
        self.dropout = DROPOUT
        self.atts = [GraphAttentionLayer(nfeat, nhid, alpha, True) for _ in range(config["nheads"])]
        for i, attention in enumerate(self.atts):
            self.add_module(f'attention_{i}', attention)
        self.out_att = GraphAttentionLayer(nhid * config["nheads"], nclass, alpha, concat=True)
        self.softmax = nn.Softmax(dim=0)
    def forward(self, feats, edge, adj):
        """forward"""
        feats = F.dropout(feats, self.dropout, training=self.training)
        feats = torch.cat([att(feats, edge, adj) for att in self.atts], dim=1)
        feats = F.dropout(feats, self.dropout, training=self.training)
        feats = F.elu(self.out_att(feats, edge, adj))
        return self.softmax(feats)
    def __str__(self):
        return self.__class__.__name__

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

class ErnieGAT(nn.Module):
    """BERT-GATES model"""
    def __init__(self, pretrained_model='nghuyong/ernie-2.0-en', nb_class=1):
        super(ErnieGAT, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = nn.Linear(self.feat_dim, nb_class)
        self.hidden_layer = config["hidden_layer"]
        self.nheads = config["nheads"]
        self.dropout = config["dropout"]
        self.weighted_adjacency_matrix = config["weighted_adjacency_matrix"]
        self.gat = GAT(nfeat=self.feat_dim, nhid=self.hidden_layer, nclass=nb_class, alpha=0.2)
    def forward(self, adj, input_ids, attention_mask, token_type_ids):
        """forward"""
        outputs = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        features = outputs.last_hidden_state
        print(features)
        print(features.shape)
        #cls_logit = self.classifier(outputs.pooler_output)
        #cls_logit = self.softmax(cls_logit)
        #cls_feats = self.bert_model(input_ids, input_mask)[0][:, 0]
        #features = self.classifier(cls_feats)
        #cls_pred = nn.Softmax(dim=0)(cls_logit)
        edge = adj.data
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = UTILS.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        features = UTILS.normalize_features(features)
        features = torch.FloatTensor(np.array(features))
        print(features.shape)
        edge = torch.FloatTensor(np.array(edge)).unsqueeze(1)
        logits = self.gat(features, edge, adj)
        #pred = 
        return logits
    def __str__(self):
        return self.__class__.__name__
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
        graph_r = GraphRepresentation(ds_name)
        if ds_name == "dbpedia":
            MAX_LENGTH = 34
        else:
            MAX_LENGTH = 32
        if mode == "train":
            for topk in config["topk"]:
                dataset = ESBenchmark(ds_name, file_n, topk, is_weighted_adjacency_matrix)
                train_data, valid_data = dataset.get_training_dataset()
                for fold in range(5):
                    fold = fold
                    print("")
                    print(f"Fold: {fold+1}, total entities: {len(train_data[fold][0])}", f"topk: top{topk}")
                    ernie_models_path = os.path.join("models", f"ernie_checkpoint-{ds_name}-{topk}-{fold}")
                    if bool(strtobool(best_epoch)) is True:
                        checkpoint = torch.load(os.path.join(ernie_models_path, f"checkpoint_best_{fold}.pt"))
                    else:
                        checkpoint = torch.load(os.path.join(ernie_models_path, f"checkpoint_latest_{fold}.pt"))
                    model = ErnieGAT()
                    model.bert_model.load_state_dict(checkpoint['bert_model'])
                    model.classifier.load_state_dict(checkpoint['classifier']) 
                    model.to(DEVICE)
                    param_optimizer = list(model.named_parameters())
                    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                    optimizer_grouped_parameters = [
                        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                        ]
                    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
                    models_path = os.path.join("models", f"ernie_gates_checkpoint-{ds_name}-{topk}-{fold}")
                    models_dir = os.path.join(os.getcwd(), models_path)
                    train(model, optimizer, train_data[fold][0], valid_data[fold][0], dataset, topk, fold, models_dir, graph_r, MAX_LENGTH)
        elif mode == "test":
            for topk in config["topk"]:
                dataset = ESBenchmark(ds_name, file_n, topk, is_weighted_adjacency_matrix)
                test_data = dataset.get_testing_dataset()
                for fold in range(5):
                    print("")
                    print(f"fold: {fold+1}, total entities: {len(test_data[fold][0])}", f"topk: top{topk}")
                    models_path = os.path.join("models", f"ernie_gates_checkpoint-{ds_name}-{topk}-{fold}")
                    model = ErnieGAT()
                    if bool(strtobool(best_epoch)) is True:
                        checkpoint = torch.load(os.path.join(models_path, f"checkpoint_best_{fold}.pt"))
                    else:
                        checkpoint = torch.load(os.path.join(models_path, f"checkpoint_latest_{fold}.pt"))
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.bert_model.load_state_dict(checkpoint['bert_model'])
                    model.classifier.load_state_dict(checkpoint['classifier'])
                    model.to(DEVICE)
                    generated_entity_summaries(model, test_data[fold][0], dataset, topk, graph_r, MAX_LENGTH)
                evaluation(dataset, topk)
def train(model, optimizer, train_data, valid_data, dataset, topk, fold, models_dir, graph_r, max_length):
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
            adj = graph_r.build_graph(triples, literal, eid)
            features = UTILS.convert_to_features(literal, TOKENIZER, max_length, triples, labels)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            target_tensor = UTILS.tensor_from_weight(len(triples), triples, labels)
            output_tensor = model(adj, all_input_ids, all_input_mask, all_segment_ids)
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
                adj = graph_r.build_graph(triples, literal, eid)
                features = UTILS.convert_to_features(literal, TOKENIZER, max_length, triples, labels)
                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
                target_tensor = UTILS.tensor_from_weight(len(triples), triples, labels)
                optimizer.zero_grad()
                output_tensor = model(adj, all_input_ids, all_input_mask, all_segment_ids)
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
                "model_state_dict": model.state_dict(),
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
                "model_state_dict": model.state_dict(),
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
def generated_entity_summaries(model, test_data, dataset, topk, graph_r, max_length):
    """"Generated entity summaries"""
    model.eval()
    with torch.no_grad():
        for eid in test_data:
            triples = dataset.get_triples(eid)
            literal = dataset.get_literals(eid)
            labels = dataset.prepare_labels(eid)
            adj = graph_r.build_graph(triples, literal, eid)
            features = UTILS.convert_to_features(literal, TOKENIZER, max_length, triples, labels)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            target_tensor = UTILS.tensor_from_weight(len(triples), triples, labels)
            output_tensor = model(adj, all_input_ids, all_input_mask, all_segment_ids)
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
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='BERT-GATES')
    PARSER.add_argument("--mode", type=str, default="test", help="mode type: train/test/all")
    PARSER.add_argument("--best_epoch", type=str, default="True", help="")
    ARGS = PARSER.parse_args()
    main(ARGS.mode, ARGS.best_epoch)
    