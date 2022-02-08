#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:56:30 2022

@author: asep
"""
import os
import numpy as np

class PredicateScore:
    """To calculate predicate value"""
    def __init__(self, triples, nodes_dict):
        self.triples = triples
        self.nodes_dict = nodes_dict
        self.in_hare = "data_inputs/hare"
        self.in_pagerank = "data_inputs/pagerank"
    @staticmethod
    def build_resources_dict(db_path, ds_name, num, method):
        """Build dictionary"""
        resources_dict = {}
        with open(os.path.join(db_path, f"{ds_name}", f"results_resources_{num}_desc_{method}.txt"), encoding="utf8") as reader:
            for line in reader:
                values = line.split("\t")
                resource = values[0]
                score = float(values[1].replace("\n", ""))
                resources_dict[resource]=score
        return resources_dict
    def get_tf_idf_scores(self):
        """Get TF-IDF score"""
        po_freq = {}
        scores = []
        for subject, predicate, _ in self.triples:
            if (subject, predicate) not in po_freq:
                po_freq[(subject, predicate)] = 1
            else:
                n_po = po_freq[(subject, predicate)]
                po_freq[(subject, predicate)] = n_po+1
            nqu=0
            for key in po_freq:
                nqu += po_freq[key]
            term_freq = po_freq[(subject, predicate)]/nqu
            freq_pred = 0
            for sub, pred, obj in self.triples:
                if (sub, predicate) == (sub, pred) or (predicate, obj) == (pred, obj):
                    freq_pred +=1
            idf = np.log(len(self.nodes_dict)/freq_pred)
            tfidf = np.multiply(term_freq, idf)
            scores.append(tfidf)
        return scores
    def get_hare_scores(self, ds_name, num):
        """GET HARE score"""
        scores = []
        resources_dict = self.build_resources_dict(self.in_hare, ds_name, num, "HARE")
        for _, pred, _ in self.triples:
            scores.append(resources_dict[pred])
        return scores
    def get_pagerank_scores(self, ds_name, num):
        """GET paegerank score"""
        scores = []
        resources_dict = self.build_resources_dict(self.in_pagerank, ds_name, num, "PAGERANK")
        for _, pred, _ in self.triples:
            scores.append(resources_dict[pred])
        return scores
    