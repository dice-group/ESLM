#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:30:53 2022

@author: asep
"""

import argparse
from tqdm import tqdm
from classes.dataset import ESBenchmark
from config import config
from classes.helpers import Utils
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
UTILS = Utils()
def main(mode, text_embed, kg_embed):
    print(config)    
    dataset = ESBenchmark("dbpedia", 6, 5, False)
    train_data, valid_data = dataset.get_training_dataset()
    max_length = 0
    for fold in range(5):
        triples = train_data[fold][0]
        for eid in triples:
            literals = dataset.get_literals(eid)
            for i, (sub_literal, pred_literal, obj_literal) in enumerate(literals):
                tokens_a = tokenizer.tokenize(sub_literal)
                tokens_b = tokenizer.tokenize(pred_literal)
                tokens_c = tokenizer.tokenize(obj_literal)
                tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
                segment_ids = [0] * len(tokens)
                if tokens_b:
                    tokens += tokens_b + ["[SEP]"]
                    segment_ids += [1] * (len(tokens_b) + 1)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                len_input_ids = len(input_ids)
                if len_input_ids > max_length:
                    max_length = len_input_ids
    for fold in range(5):
        triples = valid_data[fold][0]
        for eid in triples:
            for i, (sub_literal, pred_literal, obj_literal) in enumerate(literals):
                tokens_a = tokenizer.tokenize(sub_literal)
                tokens_b = tokenizer.tokenize(pred_literal)
                tokens_c = tokenizer.tokenize(obj_literal)
                tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
                segment_ids = [0] * len(tokens)
                if tokens_b:
                    tokens += tokens_b + ["[SEP]"]
                    segment_ids += [1] * (len(tokens_b) + 1)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                len_input_ids = len(input_ids)
                if len_input_ids > max_length:
                    max_length = len_input_ids
    test_data = dataset.get_testing_dataset()
    for fold in range(5):
        triples = test_data[fold][0]
        for eid in triples:
            for i, (sub_literal, pred_literal, obj_literal) in enumerate(literals):
                tokens_a = tokenizer.tokenize(sub_literal)
                tokens_b = tokenizer.tokenize(pred_literal)
                tokens_c = tokenizer.tokenize(obj_literal)
                tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
                segment_ids = [0] * len(tokens)
                if tokens_b:
                    tokens += tokens_b + ["[SEP]"]
                    segment_ids += [1] * (len(tokens_b) + 1)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                len_input_ids = len(input_ids)
                if len_input_ids > max_length:
                    max_length = len_input_ids
    print(f"Max length: {max_length}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GATES: Graph Attention Networks for Entity Summarization')
    parser.add_argument("--mode", type=str, default="test", help="use which mode type: train/test/all")
    parser.add_argument("--kg_embed", type=str, default="ComplEx", help="use ComplEx/DistMult/ConEx")
    parser.add_argument("--save_every", type=int, default=1, help="save model in every n epochs")
    parser.add_argument("--text_embed", type=str, default="Glove", help="use which word embedding model: fasttext/Glove")
    parser.add_argument("--word_emb_calc", type=str, default="AVG", help="use which method to compute textual form: SUM/AVG")
    
    args = parser.parse_args()
    main(args.mode, args.text_embed, args.kg_embed)

