#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:04:58 2020
"""

# Define model
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse as sp
from transformers import BertModel, BertPreTrainedModel

from config import config
from helpers import Utils

UTILS = Utils()
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
    def _prepare_attentional_mechanism_input_with_edge_features(self, w_h_nodes, weight_edges):
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
        for i, attention in enumerate(self.attentions):
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

class BERTGATES(BertPreTrainedModel):
    """BERT-GATES model"""
    def __init__(self, bert_config):
        super().__init__(bert_config, config)
        self.bert = BertModel(bert_config)
        self.input_size = 12288
        self.hidden_layer = config["hidden_layer"]
        self.nheads = config["nheads"]
        self.dropout = config["dropout"]
        self.weighted_adjacency_matrix = config["weighted_adjacency_matrix"]
        self.gat = GAT(nfeat=self.input_size, nhid=self.hidden_layer, nclass=1, alpha=0.2)
    def forward(self, adj, input_ids, segment_ids=None, input_mask=None):
        """forward"""
        facts_encode = self.bert(input_ids, segment_ids, input_mask)
        facts_encode = facts_encode[0]#torch.transpose(facts_encode[0], 0, 1)
        feats = torch.flatten(facts_encode, start_dim=1)
        edge = adj.data
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = UTILS.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        features = feats
        features = UTILS.normalize_features(features.detach().numpy())
        features = torch.FloatTensor(np.array(features))
        edge = torch.FloatTensor(np.array(edge)).unsqueeze(1)
        logits = self.gat(features, edge, adj)
        return logits
    def __str__(self):
        return self.__class__.__name__
    