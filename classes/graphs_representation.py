#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:34:40 2022

@author: asep
"""
import numpy as np
import scipy.sparse as sp
from config import config
from classes.predicatescore import PredicateScore

class GraphRepresentation:
    """To produce graph representation"""
    def __init__(self, ds_name):
        self.ds_name = ds_name
        self.weighted_edges_model = config["weighted_score"]
    def __str__(self):
        return self.__class__.__name__
    @staticmethod
    def set_nodes_dict(literal):
        """"Buid dictionary"""
        subject_list = []
        object_list = []
        for sub_literal, _, obj_literal in literal:
            subject_list.append(sub_literal)
            object_list.append(obj_literal)
        subjects = subject_list
        objects = object_list
        nodes = subjects + objects
        nodes_dict = {}
        for node in nodes:
            if node not in nodes_dict:
                nodes_dict[node] = len(nodes_dict)
        return nodes_dict
    @staticmethod
    def set_rel_dict(triples):
        """Build relation dictionary"""
        relation_list = []
        relations = relation_list
        rel_dict = {}
        for triple in triples:
            _, pred, _ = triple
            relation_list.append(pred)
        for relation in relations:
            if relation not in rel_dict:
                rel_dict[relation] = len(rel_dict)
        return rel_dict
    def build_graph(self, triples, literal, num):
        """Build graph"""
        #print(num)
        nodes_dict = self.set_nodes_dict(literal)
        rel_dict = self.set_rel_dict(triples)
        triples_idx = []
        triples_list = []
        for i, triple in enumerate(triples):
            _, pred, obj = triple
            sub_literal, _, obj_literal = literal[i]
            triples = (sub_literal, pred, obj)
            triple_tuple_idx = (nodes_dict[sub_literal], rel_dict[pred], nodes_dict[obj_literal])
            triples_idx.append(triple_tuple_idx)
            triples_list.append(triples)
        triples_idx = np.array(triples_idx)
        scores = PredicateScore(triples_list, nodes_dict)
        if self.weighted_edges_model == "tf-idf":
            weighted_values = scores.get_tf_idf_scores()
        elif self.weighted_edges_model == "hare":
            weighted_values = scores.get_hare_scores(self.ds_name, num)
        elif self.weighted_edges_model == "pagerank":
            weighted_values = scores.get_pagerank_scores(self.ds_name, num)
        else:
            weighted_values = np.ones(triples_idx.shape[0])
        adj = sp.coo_matrix((weighted_values, (triples_idx[:, 0], triples_idx[:, 2])),
                            shape=(triples_idx.shape[0], triples_idx.shape[0]),
                            dtype=np.float32)
        return adj
    