#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:30:53 2022

@author: asep
"""

import argparse
from tqdm import tqdm
from dataset import ESBenchmark
from config import config
from train_model import train_iter
from helpers import Utils

def main(mode, text_embed, kg_embed):
    print(config)
    #word_emb = utils.get_embeddings(text_embed)
        
    if config["regularization"]==True:
        weight_decay=config["weight_decay"]
    else:
        weight_decay=0
    utils = Utils()
    for ds_name in config["ds_name"]:
        if ds_name == "dbpedia":
            db_start, db_end = [1, 141], [101, 166]
        elif ds_name == "lmdb":
            db_start, db_end = [101, 166], [141, 176]
        elif ds_name == "faces":
            db_start, db_end = [1, 26], [26, 51]
        else:
           raise ValueError("The database's name must be dbpedia or lmdb or faces")
           
        dataset = ESBenchmark()
        
        entity2vec, pred2vec, entity2ix, pred2ix = utils.load_kg_embed(ds_name, kg_embed)
        entity_dict = entity2vec
        pred_dict = pred2vec
        pred2ix_size = len(pred2ix)
        entity2ix_size = len(entity2ix)
        
        if mode=="train":
            train_iter(ds_name, entity_dict, pred_dict, entity2ix_size, pred2ix_size, text_embed)
            
        #for i in tqdm(range(db_start[0], db_end[0])):
        #    triples = dataset._get_triples(ds_name, i)
        #    literals = dataset.get_literals(ds_name, i)
        #    adj = graph.build_graph(triples, literals, config["weighted_score"], ds_name, i)
            
        
        #for i in tqdm(range(db_start[1], db_end[1])):
        #    triples = dataset._get_triples(ds_name, i)
        #    literals = dataset.get_literals(ds_name, i)
        #    adj = graph.build_graph(triples, literals, config["weighted_score"], ds_name, i)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GATES: Graph Attention Networks for Entity Summarization')
    parser.add_argument("--mode", type=str, default="test", help="use which mode type: train/test/all")
    parser.add_argument("--kg_embed", type=str, default="ComplEx", help="use ComplEx/DistMult/ConEx")
    parser.add_argument("--save_every", type=int, default=1, help="save model in every n epochs")
    parser.add_argument("--text_embed", type=str, default="Glove", help="use which word embedding model: fasttext/Glove")
    parser.add_argument("--word_emb_calc", type=str, default="AVG", help="use which method to compute textual form: SUM/AVG")
    
    args = parser.parse_args()
    main(args.mode, args.text_embed, args.kg_embed)

