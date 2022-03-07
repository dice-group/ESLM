#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:04:37 2022

@author: asep
"""
import os

class TripleScoring():
    """To get triples scoring"""
    def __init__(self):
        self.in_hare = "data_inputs/hare"
        self.in_pagerank = "data_inputs/pagerank"
    @staticmethod
    def get_triple_scores(db_path, ds_name, num, method):
        """Get triples score"""
        triples_dict = {}
        with open(os.path.join(db_path, f"{ds_name}", f"results_triples_{num}_desc_{method}.txt"), encoding="utf8") as reader:
            for line in reader:
                print(line)
                values = line.split("\t")
                triple = values[0].strip()
                score = float(values[1].replace("\n", ""))
                triples_dict[triple]=score
        return triples_dict
    def get_hare_triple_scores(self, ds_name, num):
        """GET triples score for HARE"""
        triples_dict = self.get_triple_scores(self.in_hare, ds_name, num, "HARE")
        return triples_dict
    def get_pagerank_triple_scores(self, ds_name, num):
        """GET triples score for PAGERANK"""
        triples_dict = self.get_triple_scores(self.in_pagerank, ds_name, num, "PAGERANK")
        return triples_dict