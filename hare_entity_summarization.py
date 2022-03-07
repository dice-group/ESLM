#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:21:06 2022

@author: asep
"""
import os
from classes.dataset import ESBenchmark
from classes.triplescoring import TripleScoring
from config import config

def writer(db_dir, dir_path, e_id, top_rank, top_k, triple_rank_list):
    '''Write triple into file'''
    with open(os.path.join(db_dir, f"{e_id}", f"{e_id}_desc.nt"), encoding="utf8") as fin:
        with open(os.path.join(dir_path, f"{e_id}_{top_rank}{top_k}.nt"), "w", encoding="utf8") as fout:
            triples_edesc = [triple for _, triple in enumerate(fin)]
            for rank in triple_rank_list:
                fout.write(triples_edesc[rank])
TRIPLE_SCORES = TripleScoring()
for ds_name in config["ds_name"]:
    print(ds_name)
    triples_dict = {}
    for topk in config["topk"]:
        dataset = ESBenchmark(ds_name, config["file_n"], topk, False)
        test_data = dataset.get_testing_dataset()
        for fold in range(5):
            for eid in test_data[fold][0]:
                print(f"####################{eid}")
                triples = dataset.get_triples(eid)
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
                outputs = TRIPLE_SCORES.get_hare_triple_scores(ds_name, eid)
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
                