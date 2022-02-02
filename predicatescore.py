#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:56:30 2022

@author: asep
"""
import numpy as np
import os

class PredicateScore(object):
    def __init__(self, triples, nodes_dict):
        self.triples = triples
        self.nodes_dict = nodes_dict 
        self.IN_HARE = "data_inputs/hare"
        self.IN_PAGERANK = "data_inputs/pagerank"
    
    def build_resources_dict(self, db_path, ds_name, num, method):
        resources_dict = {}
        with open(os.path.join(db_path, "{}".format(ds_name), "results_resources_{}_desc_{}.txt".format(num, method)), encoding="utf8") as reader:
            for i, line in enumerate(reader):
                values = line.split("\t")
                resource = values[0]
                score = float(values[1].replace("\n", ""))
                resources_dict[resource]=score
        return resources_dict
    
    def get_tf_idf_scores(self):
        predicatesObjectsFreq = {}
        scores = []
        for sub, pred, obj in self.triples:   
            if (sub, pred) not in predicatesObjectsFreq:
                predicatesObjectsFreq[(sub, pred)] = 1
            else:
                n = predicatesObjectsFreq[(sub, pred)]
                predicatesObjectsFreq[(sub, pred)] = n+1
                    
            nqu=0
            for (s, p) in predicatesObjectsFreq.keys():
                nqu += predicatesObjectsFreq[(s, p)]
            tf = predicatesObjectsFreq[(sub, pred)]/nqu
            fPredOverGraph = 0
            for _, p, o in self.triples:
                if (s, pred) == (s, p) or (pred, o) == (p, o):
                    fPredOverGraph +=1
            idf = np.log(len(self.nodes_dict)/fPredOverGraph)
            tfidf = np.multiply(tf, idf)
            scores.append(tfidf)
          
        return scores
    
    def get_hare_scores(self, ds_name, num):
        scores = []
        resources_dict = self.build_resources_dict(self.IN_HARE, ds_name, num, "HARE")
        for sub, pred, obj in self.triples:
            scores.append(resources_dict[pred])
        return scores
    
    def get_pagerank_scores(self, ds_name, num):
        scores = []
        resources_dict = self.build_resources_dict(self.IN_PAGERANK, ds_name, num, "PAGERANK")
        #for key in resources_dict.keys():
        #    print(key)
        for sub, pred, obj in self.triples:
            #print(sub, pred, obj)
            scores.append(resources_dict[pred])
        return scores
    
    
