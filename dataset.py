#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:58:42 2021

@author: asep
"""
import os
from helpers import Utils
from stop_words import get_stop_words
from nltk.corpus import stopwords
from rdflib.plugins.parsers.ntriples import NTriplesParser, Sink

class ESBenchmark(object):
    def __init__(self, file_n = 6, topk=5, weighted_adjacency_matrix=False):
        #define dataset url
        self.file_n = file_n
        self.topk = topk
        self.weighted_adjacency_matrix = weighted_adjacency_matrix
        self.value_error = "The database's name must be dbpedia, lmdb. or faces"
        self.IN_ESBM_DIR = os.path.join(os.getcwd(), "datasets/ESBM_benchmark_v1.2")
        self.IN_DBPEDIA_DIR = os.path.join(os.getcwd(), 'datataset/ESBM_benchmark_v1.2/dbpedia_data')
        self.IN_LMDB_DIR = os.path.join(os.getcwd(), 'datasets/ESBM_benchmark_v1.2/lmdb_data')
        self.IN_FACES_DIR = os.path.join(os.getcwd(), 'datasets/FACES/faces_data')
        
    def get_5fold_train_valid_test_elist(self, ds_name_str):
        if ds_name_str == "dbpedia":
            split_path = os.path.join(self.IN_ESBM_DIR, "dbpedia_split")
        elif ds_name_str == "lmdb":
            split_path = os.path.join(self.IN_ESBM_DIR, "lmdb_split")
        elif ds_name_str == "faces":
            split_path = os.path.join(self.IN_ESBM_DIR, "faces_split")
        else:
            raise ValueError(self.value_error)
        trainList, validList, testList = [],[],[]
        for i in range(5): # 5-folds
            # read split eid files
            fold_path = os.path.join(split_path, 'Fold'+str(i))
            train_eids = self.read_split(fold_path,'train')
            valid_eids = self.read_split(fold_path,'valid')
            test_eids = self.read_split(fold_path,'test')
            trainList.append(train_eids)
            validList.append(valid_eids)
            testList.append(test_eids)
        return trainList, validList, testList
    
    def read_split(self, fold_path, split_name):
        split_eids = []
        with open(os.path.join(fold_path, "{}.txt".format(split_name)),encoding='utf-8') as f:
            for line in f:
                if len(line.strip())==0:
                    continue
                eid = int(line.split('\t')[0])
                split_eids.append(eid)
        return split_eids
    
    def get_triples(self, ds_name, num):
        if ds_name == "dbpedia":
            db_path = os.path.join(self.IN_ESBM_DIR, "dbpedia_data")
        elif ds_name == "lmdb":
            db_path = os.path.join(self.IN_ESBM_DIR, "lmdb_data")
        elif ds_name == "faces":
            db_path = os.path.join(self.IN_ESBM_DIR, "faces_data")
        else:
            raise ValueError(self.value_error)
        triples = []
        class IndexSink(Sink):
            i = 0
            j = 0
            def triple(self,s,p,o):
                #parse s,p,o to dictionaries/databases
                s = s.toPython()
                p = p.toPython()
                o = o.toPython()
                triple_tuple = (s, p, o)
                triples.append(triple_tuple)
                
        triples = []
        IndexSink = IndexSink()
        parser = NTriplesParser(IndexSink)
        with open(os.path.join(db_path, "{}".format(num), "{}_desc.nt".format(num)), 'rb') as reader:
            parser.parse(reader)
        return triples
    
    def get_labels(self, ds_name, num):
        triples = self._get_triples(ds_name, num)
        utils = Utils()
        endpoint = "http://dbpedia.org/sparql"
        triples_tuple = []
        for s, p, o in triples:
            if utils.is_URI(o):
                ol = utils.get_label_of_entity(o, endpoint)
            else: 
                ol = o
            
            pl = utils.get_label_of_entity(p, endpoint)
            sl = utils.get_label_of_entity(s, endpoint)
                
            triple = (sl, pl, ol)
            triples_tuple.append(triple)
        return triples_tuple
    
    def get_literals(self, ds_name, num):
        triples_literal=[]
        with open(os.path.join(os.getcwd(), "data_inputs/literals/{}".format(ds_name), "{}_literal.txt".format(num))) as reader:
            for literal in reader:
                values = literal.split("\t")
                sl = values[0]
                pl = values[1]
                ol = values[2].replace("\n", "")
                triple_literal_tuple = (sl, pl, ol)
                triples_literal.append(triple_literal_tuple)
        return triples_literal
    
    def get_training_dataset(self, ds_name):
        train_eids, valid_eids, _ = self.get_5fold_train_valid_test_elist(ds_name) 
        train_data = []
        valid_data = []
        
        # collect training data
        for fold, eids_per_fold in enumerate(train_eids):
            train_data_perfold = []
            edesc = {}
            for eid in eids_per_fold:
                triples = self.get_triples(ds_name, eid)
                edesc[eid] = triples
            train_data_perfold.append(edesc)
            train_data.append(train_data_perfold)
        
        # collect training data
        for fold, eids_per_fold in enumerate(valid_eids):
            valid_data_per_fold = []
            edesc = {}
            for eid in eids_per_fold:
                triples = self.get_triples(ds_name, eid)
                edesc[eid] = triples
            valid_data_per_fold.append(edesc)
            valid_data.append(valid_data_per_fold)
        return train_data, valid_data
    
    
    def get_testing_dataset(self, ds_name):
        _, _, test_eids = self.get_5fold_train_valid_test_elist(ds_name) 
        test_data = list()
        
        # collect training data
        for fold, eids_per_fold in enumerate(test_eids):
            test_data_perfold = list()
            for eid in eids_per_fold:
                triples = self.get_triples(ds_name, eid)
                edesc = {}
                edesc[eid] = triples
                test_data_perfold.append(edesc)
            test_data.append(test_data_perfold)
        return test_data
    
    def get_predicates_corpus(self, ds_name):
        train_data, _ = self.get_training_dataset(ds_name)
        utils = Utils()
        pred_dict_all = {}
        for fold, data in enumerate(train_data):
            pred_dict = {}
            for t_fold in data:
                for eid in t_fold.keys():
                    for t in t_fold[eid]:
                        _, p, _ = t
                        
                        p = utils.extract_triple_to_word(p.lower())
                        
                        if p not in pred_dict:
                            pred_dict[p]=len(pred_dict)
                            
            pred_dict_all[fold] = pred_dict                
                        
        return pred_dict_all
    
    def get_sub_obj_corpus(self, ds_name):
        train_data, _ = self.get_training_dataset(ds_name)
        utils = Utils()
        D_sub = {}
        D_obj = {}
        for fold, data in enumerate(train_data):
            d_sub_fold = []
            d_obj_fold = []
            for t_fold in data:
                for eid in t_fold.keys():
                    d_sub_eid = []
                    d_obj_eid = []
                    for t in t_fold[eid]:
                        s, _, o = t   
                        s = utils.extract_triple_to_word(s.lower())
                        o = utils.extract_triple_to_word(o.lower())
                        d_sub_eid.append(s)
                        d_obj_eid.append(o)
                    sub_words = [w for w in d_sub_eid if not w in self.get_stop_words()]
                    obj_words = [w for w in d_obj_eid if not w in self.get_stop_words()]
                    d_sub_fold.append(sub_words)
                    d_obj_fold.append(obj_words)
            D_sub[fold] = d_sub_fold
            D_obj[fold] = d_obj_fold
                        
        return D_sub, D_obj
    
    def get_stop_words(self):
        stop_words = list(get_stop_words('en'))         #About 900 stopwords
        nltk_words = list(stopwords.words('english')) #About 150 stopwords
        stop_words.extend(nltk_words)
        return stop_words

# for evaluation purpose
##IN_ESBM_DIR = os.path.join('./data', 'ESBM_benchmark_v1.2')
##IN_DBPEDIA_DIR = os.path.join('./data/ESBM_benchmark_v1.2', 'dbpedia_data')
##IN_LMDB_DIR = os.path.join('./data/ESBM_benchmark_v1.2', 'lmdb_data')
##IN_FACES_DIR = os.path.join('./data/FACES', 'faces_data')

#dataset = ESBenchmarks()
#train_data, valid_data = dataset.get_training_dataset("dbpedia")
#print("Training data")
#for fold, data in enumerate(train_data):
#    print("fold", fold)
#    for t_fold in data:
#        for t in t_fold:
#            print(t)

#print("Validation data")
#for fold, data in enumerate(valid_data):
#    print("fold", fold)
#    for t_fold in data:
#        print(len(t_fold))
#triples = dataset._get_triples("dbpedia", 1)
#for triple in triples:
#    print(triple)
##train, valid, test = dataset.get_5fold_train_valid_test_elist("dbpedia", IN_ESBM_DIR)
##print(len(train))

    
    