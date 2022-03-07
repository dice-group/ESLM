#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:16:02 2022

@author: asep
"""
from tqdm import tqdm
import numpy as np
from HARE.Utility.getTransitionMatrices import getTransitionMatrices
from HARE.Utility.parseRDF import parseRDF
from HARE.Computations.hare import hare
from HARE.Computations.pagerank import pagerank

REPETITIONS = 5
DSNAME = "lmdb"

def run_hare(data, ds_name, index):
    """Run HARE"""
    print("WITH: ", data)
    parseRDF(data, index, ds_name)
    getTransitionMatrices(data, ds_name)
    print(".....HARE.....")
    runtimes_hare = np.array(REPETITIONS*[.0])
    for i in range(REPETITIONS-1):
        runtime = hare(data, ds_name, epsilon=10**(-2), damping=.85, saveresults=True, printerror=False, printruntimes=True)
        runtimes_hare[i] = runtime
    print("Average Runtime HARE: ", np.mean(runtimes_hare))
    print(".....PAGERANK.....")
    runtimes_pagerank = np.array(REPETITIONS*[.0])
    for i in range(REPETITIONS):
        runtime = pagerank(data, ds_name, epsilon=10**(-3), damping=.85, saveresults=True, printerror=False, printruntimes=True)
        runtimes_pagerank[i] = runtime
    print("Average Runtime PAGERANK: ", np.mean(runtimes_pagerank))
if DSNAME == "dbpedia":
    DB_START, DB_END = [1, 141], [101, 166]
elif DSNAME == "lmdb":
    DB_START, DB_END = [101, 166], [141, 176]
elif DSNAME == "faces":
    DB_START, DB_END = [1, 26], [26, 51]
else:
    raise ValueError("The database's name must be dbpedia or lmdb or faces")
for eid in tqdm(range(DB_START[0], DB_END[0])):
    idx = eid
    triples = f"{idx}_desc.nt"
    run_hare(triples, DSNAME, idx)
for eid in tqdm(range(DB_START[1], DB_END[1])):
    idx = eid
    triples = f"{idx}_desc.nt"
    run_hare(triples, DSNAME, idx)
    