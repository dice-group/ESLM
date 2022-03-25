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
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LENGTH = 39
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

class BertGATES(nn.Module):
    """BERT-GATES model"""
    def __init__(self, pretrained_model='bert-base-uncased', nb_class=1):
        super(BertGATES, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = nn.Linear(self.feat_dim, nb_class)
        self.hidden_layer = config["hidden_layer"]
        self.nheads = config["nheads"]
        self.dropout = config["dropout"]
        self.weighted_adjacency_matrix = config["weighted_adjacency_matrix"]
        self.gat = GAT(nfeat=self.feat_dim, nhid=self.hidden_layer, nclass=nb_class, alpha=0.2)
    def forward(self, adj, input_ids, input_mask=None):
        """forward"""
        cls_feats = self.bert_model(input_ids, input_mask)[0][:, 0]
        features = cls_feats
        #features = self.classifier(cls_feats)
        #cls_pred = nn.Softmax(dim=0)(cls_logit)
        edge = adj.data
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = UTILS.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        #features = UTILS.normalize_features(features.detach().numpy())
        #features = torch.FloatTensor(np.array(features))
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
def main(mode):
    """Main module"""
    file_n = config["file_n"]
    is_weighted_adjacency_matrix = config["weighted_adjacency_matrix"]
    if mode == "train":
        log_file_path = os.path.join(os.getcwd(), 'logs/BertGATES_log.txt')
        with open(log_file_path, 'w', encoding="utf-8") as log_file:
            pass
    for ds_name in config["ds_name"]:
        graph_r = GraphRepresentation(ds_name)
        if mode == "test":
            for topk in config["topk"]:
                filename = 'logs/BertGATES_log.txt'
                use_epoch = UTILS.read_epochs_from_log(ds_name, topk, filename)
                dataset = ESBenchmark(ds_name, file_n, topk, is_weighted_adjacency_matrix)
                test_data = dataset.get_testing_dataset()
                fmeasure_scores = []
                ndcg_scores = []
                map_scores = []
                models = []
                for fold in range(5):
                    models_path = os.path.join("models", f"bert_gates_checkpoint-{ds_name}-{topk}-{fold}")
                    model = BertGATES()
                    checkpoint = torch.load(os.path.join(models_path, f"checkpoint_epoch_{use_epoch[fold]}.pt"))
                    model.bert_model.load_state_dict(checkpoint["bert_model"])
                    model.classifier.load_state_dict(checkpoint["classifier"])
                    model.to(DEVICE)
                    models.append(model)
                    
                for fold in range(5):
                    print("")
                    print(f"fold: {fold+1}, total entities: {len(test_data[fold][0])}", f"topk: top{topk}")
                    fmeasure_score, ndcg_score, map_score = generated_entity_summaries(models, test_data[fold][0], dataset, topk, graph_r, fold)
                    fmeasure_scores.append(fmeasure_score)
                    ndcg_scores.append(ndcg_score)
                    map_scores.append(map_score)
                print(f"{dataset.ds_name}@top{topk}: F-Measure={np.average(fmeasure_scores)}, NDCG={np.average(ndcg_scores)}, MAP={np.average(map_scores)}")

def generated_entity_summaries(models, test_data, dataset, topk, graph_r, fold):
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
            adj = graph_r.build_graph(triples, literal, eid)
            features = UTILS.convert_to_features_with_subject(literal, TOKENIZER, MAX_LENGTH, triples, labels)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            #all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            target_tensor = UTILS.tensor_from_weight(len(triples), triples, labels)
            output_tensor = evaluate_n_members(models, fold, all_input_ids, all_input_mask)
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
# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, fold, all_input_ids, all_input_mask):
    if fold==4:
        subset = [members[0],  members[4]]
    else:
        subset = [members[fold],  members[fold+1]]
    yhat = ensemble_predictions(subset, all_input_ids, all_input_mask)
    return yhat

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, all_input_ids, all_input_mask):
	# make predictions
    yhats = torch.stack([model(all_input_ids, all_input_mask) for model in members])
    result = torch.sum(yhats, axis=0)
    return result
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='BERT-GATES')
    PARSER.add_argument("--mode", type=str, default="test", help="mode type: train/test/all")
    ARGS = PARSER.parse_args()
    main(ARGS.mode)
    