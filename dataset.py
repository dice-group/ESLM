#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:58:42 2021

@author: asep
"""
import os
from nltk.corpus import stopwords
from rdflib.plugins.parsers.ntriples import NTriplesParser, Sink
from stop_words import get_stop_words
from helpers import Utils
from config import config

utils = Utils()

class ESBenchmark:
    """This class contains modules pertaining to dataset processes"""
    def __init__(self, file_n = 6, topk=5, weighted_adjacency_matrix=False):
        self.file_n = file_n
        self.topk = topk
        self.weighted_adjacency_matrix = weighted_adjacency_matrix
        self.value_error = "The database's name must be dbpedia, lmdb. or faces"
        self.in_esbm_dir = os.path.join(os.getcwd(), "datasets/ESBM_benchmark_v1.2")
        self.in_faces_dir = os.path.join(os.getcwd(), 'datasets/FACES/faces_data')
    def get_5fold_train_valid_test_elist(self, ds_name_str):
        """Get splitted data including train, valid, and test data"""
        if ds_name_str == "dbpedia":
            split_path = os.path.join(self.in_esbm_dir, "dbpedia_split")
        elif ds_name_str == "lmdb":
            split_path = os.path.join(self.in_esbm_dir, "lmdb_split")
        elif ds_name_str == "faces":
            split_path = os.path.join(self.in_esbm_dir, "faces_split")
        else:
            raise ValueError(self.value_error)
        train_data, valid_data, test_data = [],[],[]
        for i in range(5): # 5-folds
            # read split eid files
            fold_path = os.path.join(split_path, 'Fold'+str(i))
            train_eids = self.read_split(fold_path,'train')
            valid_eids = self.read_split(fold_path,'valid')
            test_eids = self.read_split(fold_path,'test')
            train_data.append(train_eids)
            valid_data.append(valid_eids)
            test_data.append(test_eids)
        return train_data, valid_data, test_data
    def read_split(self, fold_path, split_name):
        """Read data from splitted txt"""
        split_eids = []
        with open(os.path.join(fold_path, f"{split_name}.txt"),encoding='utf-8') as reader:
            for line in reader:
                if len(line.strip())==0:
                    continue
                eid = int(line.split('\t')[0])
                split_eids.append(eid)
        return split_eids
    def get_triples(self, ds_name, num):
        """Get triples"""
        if ds_name == "dbpedia":
            db_path = os.path.join(self.in_esbm_dir, "dbpedia_data")
        elif ds_name == "lmdb":
            db_path = os.path.join(self.in_esbm_dir, "lmdb_data")
        elif ds_name == "faces":
            db_path = os.path.join(self.in_esbm_dir, "faces_data")
        else:
            raise ValueError(self.value_error)
        triples = []
        class IndexSink(Sink):
            i = 0
            j = 0
            def triple(self,sub,pred,obj):
                sub = sub.toPython()
                pred = pred.toPython()
                obj = obj.toPython()
                triple_tuple = (sub, pred, obj)
                triples.append(triple_tuple)
        triples = []
        index_sink = IndexSink()
        parser = NTriplesParser(index_sink)
        with open(os.path.join(db_path,f"{num}",f"{num}_desc.nt"),'rb') as reader:
            parser.parse(reader)
        return triples
    def get_labels(self, ds_name, num):
        """Get entity label from knowledge base"""
        triples = self.get_triples(ds_name, num)
        helpers = Utils()
        endpoint = "http://dbpedia.org/sparql"
        triples_tuple = []
        for sub, pred, obj in triples:
            if utils.is_URI(obj):
                obj_literal = helpers.get_label_of_entity(obj, endpoint)
            else:
                obj_literal = obj
            pred_literal = helpers.get_label_of_entity(pred, endpoint)
            sub_literal = helpers.get_label_of_entity(sub, endpoint)
            triple = (sub_literal, pred_literal, obj_literal)
            triples_tuple.append(triple)
        return triples_tuple
    def get_literals(self, ds_name, num):
        """Get literal value from literal txt"""
        triples_literal=[]
        with open(os.path.join(os.getcwd(), f"data_inputs/literals/{ds_name}", f"{num}_literal.txt"), encoding="utf-8") as reader:
            for literal in reader:
                values = literal.split("\t")
                sub_literal = values[0]
                pred_literal = values[1]
                obj_literal = values[2].replace("\n", "")
                triple_literal_tuple = (sub_literal, pred_literal, obj_literal)
                triples_literal.append(triple_literal_tuple)
        return triples_literal
    def get_training_dataset(self, ds_name):
        """Get all training dan validation data"""
        train_eids, valid_eids, _ = self.get_5fold_train_valid_test_elist(ds_name)
        train_data = []
        valid_data = []
        for eids_per_fold in train_eids:
            train_data_perfold = []
            edesc = {}
            for eid in eids_per_fold:
                triples = self.get_triples(ds_name, eid)
                edesc[eid] = triples
            train_data_perfold.append(edesc)
            train_data.append(train_data_perfold)
        for eids_per_fold in valid_eids:
            valid_data_per_fold = []
            edesc = {}
            for eid in eids_per_fold:
                triples = self.get_triples(ds_name, eid)
                edesc[eid] = triples
            valid_data_per_fold.append(edesc)
            valid_data.append(valid_data_per_fold)
        return train_data, valid_data
    def get_testing_dataset(self, ds_name):
        """Get all testing data"""
        _, _, test_eids = self.get_5fold_train_valid_test_elist(ds_name)
        test_data = []
        for eids_per_fold in test_eids:
            test_data_perfold = []
            for eid in eids_per_fold:
                triples = self.get_triples(ds_name, eid)
                edesc = {}
                edesc[eid] = triples
                test_data_perfold.append(edesc)
            test_data.append(test_data_perfold)
        return test_data
    def prepare_labels(self, ds_name, num, top_n, file_n):
        """Create gold label dictionary from gold summary triples"""
        if ds_name == "dbpedia":
            db_path = os.path.join(self.in_esbm_dir, "dbpedia_data")
        elif ds_name == "lmdb":
            db_path = os.path.join(self.in_esbm_dir, "lmdb_data")
        elif ds_name == "faces":
            db_path = os.path.join(self.in_esbm_dir, "faces_data")
        else:
            raise ValueError(self.value_error)
        per_entity_label_dict = {}
        class IndexSink(Sink):
            """Indexing triples"""
            i = 0
            j = 0
            def triple(self,sub,pred,obj):
                """Get triples"""
                sub = sub.toPython()
                pred = pred.toPython()
                obj = obj.toPython()
                triple_tuple = (sub, pred, obj)
                triples.append(triple_tuple)
        IndexSink = IndexSink()
        for i in range(file_n):
            triples = []
            parser = NTriplesParser(IndexSink)
            with open(os.path.join(db_path, f"{num}", f"{num}_gold_top{top_n}_{i}.nt"), 'rb') as reader:
                parser.parse(reader)
            for _, pred, obj in triples:
                utils.counter(per_entity_label_dict, "{}++$++{}".format(pred, obj))
        return per_entity_label_dict
    def get_stop_words(self):
        """Get stop words based on NLTK library"""
        stop_words = list(get_stop_words('en'))         #About 900 stopwords
        nltk_words = list(stopwords.words('english')) #About 150 stopwords
        stop_words.extend(nltk_words)
        return stop_words
    def get_gold_summaries(self, ds_name, num, topk):
        """Get all triples from gold summary"""
        if ds_name == "dbpedia":
            db_path = os.path.join(self.in_esbm_dir, "dbpedia_data")
        elif ds_name == "lmdb":
            db_path = os.path.join(self.in_esbm_dir, "lmdb_data")
        elif ds_name == "faces":
            db_path = os.path.join(self.in_esbm_dir, "faces_data")
        else:
            raise ValueError(self.value_error)
        triples_dict = {}
        triples = self.get_triples(ds_name, num)
        for triple in triples:
            if triple not in triples_dict:
                triples_dict[triple] = len(triples_dict)
        triples_summary = []
        class IndexSink(Sink):
            """indexing triples"""
            i = 0
            j = 0
            def triple(self,sub,pred,obj):
                """Get triples"""
                sub = sub.toPython()
                pred = pred.toPython()
                obj = obj.toPython()
                triple_tuple = (sub, pred, obj)
                triples_summary.append(triple_tuple)
        gold_summary_list = []
        index_sink = IndexSink()
        parser = NTriplesParser(index_sink)
        for i in range(config["file_n"]):
            with open(os.path.join(db_path, f"{num}", f"{num}_gold_top{topk}_{i}.nt"), 'rb') as reader:
                parser.parse(reader)
            n_list = []
            for triple in triples_summary:
                gold_id = triples_dict[triple]
                n_list.append(gold_id)
            gold_summary_list.append(n_list)
        return gold_summary_list
    