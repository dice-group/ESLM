#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:16:02 2022

@author: asep
"""

from HARE.Utility.getTransitionMatrices import getTransitionMatrices
from HARE.Utility.parseRDF import parseRDF
from HARE.Computations.hare import hare
from HARE.Computations.pagerank import pagerank
import numpy as np
from tqdm import tqdm

repetitions = 5
dsname="lmdb"

def run_hare(data, ds_name, index):
    print("WITH: ", data)
    parseRDF(data, index, ds_name)
    getTransitionMatrices(data, ds_name)
    print(".....HARE.....")
    runtimes_hare = np.array(repetitions*[.0])
    for i in range(repetitions-1):
        runtime = hare(data, ds_name, epsilon=10**(-2), damping = .85, saveresults=True, printerror=False, printruntimes=True)
        runtimes_hare[i] = runtime
    print("Average Runtime HARE: ", np.mean(runtimes_hare))
    print(".....PAGERANK.....")
    runtimes_pagerank = np.array(repetitions*[.0])
    for i in range(repetitions):
        runtime = pagerank(data, ds_name, epsilon=10**(-3), damping = .85, saveresults=True, printerror=False, printruntimes=True)
        runtimes_pagerank[i] = runtime
    print("Average Runtime PAGERANK: ", np.mean(runtimes_pagerank))
    
if dsname == "dbpedia":
    db_start, db_end = [1, 141], [101, 166]
elif dsname == "lmdb":
    db_start, db_end = [101, 166], [141, 176]
elif dsname == "faces":
    db_start, db_end = [1, 26], [26, 51]
else:
    raise ValueError("The database's name must be dbpedia or lmdb or faces")    
for i in tqdm(range(db_start[0], db_end[0])):
    triples = f"{i}_desc.nt"
    index = i
    run_hare(triples, dsname, index)
for i in tqdm(range(db_start[1], db_end[1])):
    triples = f"{}_desc.nt"
    index = i
    run_hare(triples, dsname, index)
    