#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:21:06 2022

@author: asep
"""

from dataset import ESBenchmark
from config import config
from triplescoring import TripleScoring
import os

def writer(db_dir, directory, eid, top_or_rank, topk, rank_list):
    with open(os.path.join(db_dir, f"{eid}", f"{eid}_desc.nt"), encoding="utf8") as fin:
        with open(os.path.join(directory, f"{eid}_{top_or_rank}{topk}.nt"), "w", encoding="utf8") as fout:
            triples = [triple for _, triple in enumerate(fin)]
            for rank in rank_list:
                fout.write(triples[rank])
        
triple_scores = TripleScoring()
for ds_name in config["ds_name"]:
    print(ds_name)
    triples_dict = {}
    for topk in config["topk"]:
        dataset = ESBenchmark(ds_name, config["file_n"], topk, False)
        test_data = dataset.get_testing_dataset()
        for fold in range(5):
            for eid in test_data[fold][0]:
                print(f"####################{eid}")
                triples  = dataset.get_triples(eid)
                triples_dict = {}
                for triple in triples:
                    sub, pred, obj = triple
                    triple_txt = f"{sub} {pred} {obj}"
                    if triple_txt not in triples_dict:
                        triples_dict[triple_txt.strip()] = len(triples_dict)
                directory = f"outputs/{ds_name}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                directory = f"outputs/{ds_name}/{eid}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                outputs = triple_scores.get_hare_triple_scores(ds_name, eid)
                rank_list = []
                print(triples_dict)
                for output in list(outputs)[:topk]:
                    print(output.strip())
                    rank_list.append(triples_dict[output])
                top_or_rank = "top"
                writer(dataset.get_db_path, directory, eid, top_or_rank, topk, rank_list)
                rank_list = []
                for output in list(outputs)[:topk]:
                    rank_list.append(triples_dict[output])
                top_or_rank = "rank_top"
                writer(dataset.get_db_path, directory, eid, top_or_rank, topk, rank_list)
                