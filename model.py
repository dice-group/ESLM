#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:04:58 2020
"""

# Define model
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

from config import config
from helpers import Utils
from transformers import BertTokenizer, BertModel, BertConfig, BertPreTrainedModel

utils = Utils()
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, weighted_adjacency_matrix, alpha, concat=True, in_edge_features=1):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.in_edge_features = in_edge_features
        self.weighted_adjacency_matrix = weighted_adjacency_matrix
        self.weight = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if weighted_adjacency_matrix==False:
            self.att = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        else:
            self.att = nn.Parameter(torch.empty(size=(3*out_features, 1)))
        nn.init.xavier_uniform_(self.att.data, gain=1.414)
        self.w_edge = nn.Parameter(torch.empty(size=(in_edge_features, out_features)))
        nn.init.xavier_uniform_(self.w_edge.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, h_nodes, edge, adj):
        w_h_nodes = torch.mm(h_nodes, self.weight) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        if self.weighted_adjacency_matrix==True:
            weight_edges = torch.mm(edge, self.w_edge)
            a_input = self._prepare_attentional_mechanism_input_with_edge_features(w_h_nodes, weight_edges)
        else:
            a_input = self._prepare_attentional_mechanism_input(w_h_nodes)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_nodes_prime = torch.matmul(attention, w_h_nodes)
        if self.concat:
            return F.elu(h_nodes_prime)
        else:
            return h_nodes_prime
    def _prepare_attentional_mechanism_input(self, w_h_nodes):
        nodes = w_h_nodes.size()[0]
        w_h_nodes_repeated_in_chunks = w_h_nodes.repeat_interleave(nodes, dim=0)
        w_h_nodes_repeated_alternating = w_h_nodes.repeat(nodes, 1)
        all_combinations_matrix = torch.cat([w_h_nodes_repeated_in_chunks, w_h_nodes_repeated_alternating], dim=1)
        return all_combinations_matrix.view(nodes, nodes, 2 * self.out_features)
    
    def _prepare_attentional_mechanism_input_with_edge_features(self, w_h_nodes, weight_edges):
        nodes = w_h_nodes.size()[0] # number of nodes
        w_h_nodes_repeated_in_chunks = w_h_nodes.repeat_interleave(nodes, dim=0)
        w_h_nodes_repeated_alternating = w_h_nodes.repeat(nodes, 1)
        weight_edges_repeated_in_chunks = weight_edges.repeat_interleave(nodes, dim=0)
        all_combinations_matrix = torch.cat([w_h_nodes_repeated_in_chunks, w_h_nodes_repeated_alternating, weight_edges_repeated_in_chunks], dim=1)
        return all_combinations_matrix.view(nodes, nodes, 3 * self.out_features)
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    """GAT model"""
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, weighted_adjacency_matrix):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout, weighted_adjacency_matrix, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout, weighted_adjacency_matrix, alpha=alpha, concat=True)
        self.softmax = nn.Softmax(dim=0)
    def forward(self, x, edge, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, edge, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, edge, adj))
        #return F.log_softmax(x, dim=1)
        return self.softmax(x)

class BERT_GATES(BertPreTrainedModel):
    """BERT-GATES model"""
    def __init__(self, bert_config, config):
        super(BERT_GATES, self).__init__(bert_config, config)
        self.bert = BertModel(bert_config)
        self.input_size = 12288
        self.hidden_layer = config["hidden_layer"]
        self.nheads = config["nheads"]
        self.dropout = config["dropout"]
        self.weighted_adjacency_matrix = config["weighted_adjacency_matrix"]
        self.gat = GAT(nfeat=self.input_size, nhid=self.hidden_layer, nclass=1, dropout=self.dropout, alpha=0.2, nheads=self.nheads, weighted_adjacency_matrix=self.weighted_adjacency_matrix)   
    def forward(self, adj, input_ids, segment_ids=None, input_mask=None):
        facts_encode = self.bert(input_ids, segment_ids, input_mask)
        facts_encode = facts_encode[0]#torch.transpose(facts_encode[0], 0, 1)
        feats = torch.flatten(facts_encode, start_dim=1)
        edge = adj.data
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = utils.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        features = feats
        features = utils.normalize_features(features.detach().numpy())
        features = torch.FloatTensor(np.array(features))
        edge = torch.FloatTensor(np.array(edge)).unsqueeze(1)
        logits = self.gat(features, edge, adj)
        return logits
    