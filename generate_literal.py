#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:15:08 2020

@author: Asep Fajar Firmansyah
"""
import os
import argparse
from tqdm import tqdm
from dataset import ESBenchmark

root_dir = os.getcwd()

def main(ds_name):
    """Main function"""
    if ds_name == "dbpedia":
        db_start, db_end = [1, 141], [101, 166]
    elif ds_name == "lmdb":
        db_start, db_end = [101, 166], [141, 176]
    elif ds_name == "faces":
        db_start, db_end = [1, 26], [26, 51]
    else:
        raise ValueError("The database's name must be dbpedia or lmdb or faces")
    dataset = ESBenchmark
    for i in tqdm(range(db_start[0], db_end[0])):
        with open(os.path.join(root_dir, f"data_inputs/literals/{ds_name}/", f"{i}_literal.txt"), "w", encoding="utf-8") as reader:
            reader.write("")
        triples_tuple = dataset.get_labels(ds_name, i)
        for sub_literal, pred_literal, obj_literal in triples_tuple:
            with open(os.path.join(root_dir, f"data_inputs/literals/{ds_name}/", f"{i}_literal.txt"), "a", encoding="utf-8") as reader:
                reader.write(f"{sub_literal}\t{pred_literal}\t{obj_literal}\n")
    for i in tqdm(range(db_start[1], db_end[1])):
        with open(os.path.join(root_dir, f"data_inputs/literals/{ds_name}/", f"{i}_literal.txt"), "w", encoding="utf-8") as reader:
            reader.write("")
        triples_tuple = dataset.get_labels(ds_name, i)
        for sub_literal, pred_literal, obj_literal in triples_tuple:
            with open(os.path.join(root_dir, f"data_inputs/literals/{ds_name}/", f"{i}_literal.txt"), "a", encoding="utf-8") as reader:
                reader.write(f"{sub_literal}\t{pred_literal}\t{obj_literal}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GATES: Converting dataset to formatted data')
    parser.add_argument("--ds_name", type=str, default="dbpedia", help="use dbpedia or lmdb")
    args = parser.parse_args()
    main(args.ds_name)
