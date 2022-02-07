#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:34:40 2022

@author: asep
"""
import numpy as np
import scipy.sparse as sp
from predicatescore import PredicateScore

class GraphRepresentation:
    """To produce graph representation"""
    def build_graph(self, triples, literals, weighted_edges_model, ds_name, num):
        """Build graph"""
        subject_list = []
        relation_list = []
        object_list = []
        triples_idx=[]
        for i, triple in enumerate(triples):
            _, pred, obj = triple
            sub_literal, _, obj_literal = literals[i]
            subject_list.append(sub_literal)
            relation_list.append(pred)
            object_list.append(obj_literal)
        relations = relation_list
        subjects = subject_list
        objects = object_list
        nodes = subjects + objects
        relations_dict = {}
        for relation in relations:
            if relation not in relations_dict:
                relations_dict[relation] = len(relations_dict)    
        nodes_dict = {}
        for node in nodes:
            if node not in nodes_dict :
                nodes_dict[node] = len(nodes_dict)
        triples_list=[]
        for i, triple in enumerate(triples):
            sub, pred, obj = triple
            sub_literal, pred_literal, obj_literal = literals[i]
            triples = (sub_literal, pred, obj)
            triple_tuple_idx = (nodes_dict[sub_literal], relations_dict[pred], nodes_dict[obj_literal])
            triples_idx.append(triple_tuple_idx)
            triples_list.append(triples)
        triples_idx = np.array(triples_idx)
        scores = PredicateScore(triples_list, nodes_dict)
        if weighted_edges_model=="tf-idf":
            weighted_values = scores.get_tf_idf_scores()
        elif weighted_edges_model=="hare":
            weighted_values = scores.get_hare_scores(ds_name, num)
        elif weighted_edges_model=="pagerank":
            weighted_values = scores.get_pagerank_scores(ds_name, num)
        else:
            weighted_values = np.ones(triples_idx.shape[0])
        adj = sp.coo_matrix((weighted_values, (triples_idx[:, 0], triples_idx[:, 2])),
                            shape=(triples_idx.shape[0], triples_idx.shape[0]),
                            dtype=np.float32)
        return adj
    