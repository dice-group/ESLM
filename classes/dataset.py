#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:58:42 2021

@author: asep
"""
import os
from rdflib.plugins.parsers.ntriples import NTriplesParser, Sink
from classes.helpers import Utils

UTILS = Utils()
class ESBenchmark:
    """This class contains modules pertaining to dataset processes"""
    def __init__(self, ds_name, file_n=6, topk=5, weighted_adjacency_matrix=False):
        self.file_n = file_n
        self.topk = topk
        self.weighted_adjacency_matrix = weighted_adjacency_matrix
        self.in_esbm_dir = os.path.join(os.getcwd(), "datasets/ESBM_benchmark_v1.2")
        self.in_faces_dir = os.path.join(os.getcwd(), 'datasets/FACES')
        self.ds_name = ds_name
        if ds_name == "dbpedia":
            self.db_path = os.path.join(self.in_esbm_dir, "dbpedia_data")
        elif ds_name == "lmdb":
            self.db_path = os.path.join(self.in_esbm_dir, "lmdb_data")
        elif ds_name == "faces":
            self.db_path = os.path.join(self.in_faces_dir, "faces_data")
        else:
            raise ValueError("The database name must be dbpedia, lmdb. or faces")
    def get_5fold_train_valid_test_elist(self, ds_name_str):
        """Get splitted data including train, valid, and test data"""
        if ds_name_str == "dbpedia":
            split_path = os.path.join(self.in_esbm_dir, "dbpedia_split")
        elif ds_name_str == "lmdb":
            split_path = os.path.join(self.in_esbm_dir, "lmdb_split")
        elif ds_name_str == "faces":
            split_path = os.path.join(self.in_esbm_dir, "faces_split")
        else:
            raise ValueError("The database name must be dbpedia, lmdb. or faces")
        train_data, valid_data, test_data = [], [], []
        for i in range(5): # 5-folds
            # read split eid files
            fold_path = os.path.join(split_path, 'Fold'+str(i))
            train_eids = self.read_split(fold_path, 'train')
            valid_eids = self.read_split(fold_path, 'valid')
            test_eids = self.read_split(fold_path, 'test')
            train_data.append(train_eids)
            valid_data.append(valid_eids)
            test_data.append(test_eids)
        return train_data, valid_data, test_data
    def get_triples(self, num):
        """Get triples"""
        triples = []
        class IndexSink(Sink):
            """Triple Indexing"""
            i = 0
            j = 0
            def __str__(self):
                return self.__class__.__name__
            @staticmethod
            def triple(sub, pred, obj):
                """Get triples"""
                sub = sub.toPython()
                pred = pred.toPython()
                obj = obj.toPython()
                triple_tuple = (sub, pred, obj)
                triples.append(triple_tuple)
        index_sink = IndexSink()
        parser = NTriplesParser(index_sink)
        with open(os.path.join(self.db_path, f"{num}", f"{num}_desc.nt"), 'rb') as reader:
            parser.parse(reader)
        return triples
    def get_labels(self, num):
        """Get entity label from knowledge base"""
        triples = self.get_triples(num)
        if self.ds_name == "dbpedia" or self.ds_name == "faces":
            endpoint = "http://dbpedia.org/sparql"
        elif self.ds_name == "lmdb":
            endpoint = "https://api.triplydb.com/datasets/Triply/linkedmdb/services/linkedmdb/sparql"
        triples_tuple = []
        for sub, pred, obj in triples:
            if UTILS.is_uri(obj) and self.ds_name == "dbpedia":
                obj_literal = UTILS.get_label_of_entity(obj, endpoint)
            elif UTILS.is_uri(obj) and self.ds_name == "lmdb":
                #obj_literal = UTILS.get_label_of_entity_lmdb("entity", obj, endpoint)
                if "dbpedia" in obj:
                    endpoint = "http://dbpedia.org/sparql"
                obj_literal = UTILS.get_label_of_entity(obj, endpoint)
            elif UTILS.is_uri(obj) and self.ds_name=="faces":
                obj_literal = UTILS.get_label_of_entity(obj, endpoint)
            else:
                if type(obj) == str:
                    if obj.isupper():
                        obj_literal = obj
                    else:
                        obj_literal = obj.title()
                else:
                    obj_literal = obj
            if self.ds_name == "dbpedia" or self.ds_name == "faces":
                pred_literal = UTILS.get_label_of_entity(pred, endpoint)
                sub_literal = UTILS.get_label_of_entity(sub, endpoint)
            elif self.ds_name == "lmdb":
                pred_literal = UTILS.get_label_of_entity_lmdb("property", pred, endpoint)
                sub_literal = UTILS.get_uri_label(sub)
            triple = (sub_literal, pred_literal, obj_literal)
            triples_tuple.append(triple)
        return triples_tuple
    def get_literals(self, num):
        """Get literal value from literal txt"""
        triples_literal = []
        path = os.path.join(os.getcwd(), f"data_inputs/literals/{self.ds_name}")
        with open(os.path.join(path, f"{num}_literal.txt"), encoding="utf-8") as reader:
            for literal in reader:
                values = literal.split("\t")
                sub_literal = values[0]
                pred_literal = values[1]
                obj_literal = values[2].replace("\n", "")
                triple_literal_tuple = (sub_literal, pred_literal, obj_literal)
                triples_literal.append(triple_literal_tuple)
        return triples_literal
    def get_training_dataset(self):
        """Get all training dan validation data"""
        train_eids, valid_eids, _ = self.get_5fold_train_valid_test_elist(self.ds_name)
        train_data = []
        valid_data = []
        for eids_per_fold in train_eids:
            train_data_perfold = []
            edesc = {}
            for eid in eids_per_fold:
                triples = self.get_triples(eid)
                edesc[eid] = triples
            train_data_perfold.append(edesc)
            train_data.append(train_data_perfold)
        for eids_per_fold in valid_eids:
            valid_data_per_fold = []
            edesc = {}
            for eid in eids_per_fold:
                triples = self.get_triples(eid)
                edesc[eid] = triples
            valid_data_per_fold.append(edesc)
            valid_data.append(valid_data_per_fold)
        return train_data, valid_data
    def get_testing_dataset(self):
        """Get all testing data"""
        _, _, test_eids = self.get_5fold_train_valid_test_elist(self.ds_name)
        test_data = []
        for eids_per_fold in test_eids:
            test_data_perfold = []
            edesc = {}
            for eid in eids_per_fold:
                triples = self.get_triples(eid)
                edesc[eid] = triples
            test_data_perfold.append(edesc)
            test_data.append(test_data_perfold)
        return test_data
    def prepare_labels(self, num):
        """Create gold label dictionary from gold summary triples"""
        per_entity_label_dict = {}
        class IndexSink(Sink):
            """Indexing triples"""
            i = 0
            j = 0
            def __str__(self):
                return self.__class__.__name__
            @staticmethod
            def triple(sub, pred, obj):
                """Get triple"""
                sub = sub.toPython()
                pred = pred.toPython()
                obj = obj.toPython()
                triple_tuple = (sub, pred, obj)
                triples.append(triple_tuple)
        index_sink = IndexSink()
        for i in range(self.file_n):
            triples = []
            parser = NTriplesParser(index_sink)
            path = os.path.join(self.db_path, f"{num}")
            with open(os.path.join(path, f"{num}_gold_top{self.topk}_{i}.nt"), 'rb') as reader:
                parser.parse(reader)
            for _, pred, obj in triples:
                UTILS.counter(per_entity_label_dict, f"{pred}++$++{obj}")
        return per_entity_label_dict
    def get_gold_summaries(self, num, triples_dict):
        """Get all triples from gold summary"""
        gold_summary_list = []
        for i in range(self.file_n):
            triples_summary = []
            class IndexSink(Sink):
                """indexing triples"""
                i = 0
                j = 0
                def __str__(self):
                    return self.__class__.__name__
                @staticmethod
                def triple(sub, pred, obj):
                    """Get triples"""
                    triple_tuple = (sub.toPython(), pred.toPython(), obj.toPython())
                    triples_summary.append(triple_tuple)
            path = os.path.join(self.db_path, f"{num}")
            index_sink = IndexSink()
            parser = NTriplesParser(index_sink)
            with open(os.path.join(path, f"{num}_gold_top{self.topk}_{i}.nt"), 'rb') as reader:
                parser.parse(reader)
            n_list = []
            for triple in triples_summary:
                gold_id = triples_dict[triple]
                n_list.append(gold_id)
        gold_summary_list.append(n_list)
        return gold_summary_list
    def triples_dictionary(self, num):
        """Build triple dictionary"""
        triples_dict = {}
        triples = self.get_triples(num)
        for triple in triples:
            if triple not in triples_dict:
                triples_dict[triple] = len(triples_dict)
        return triples_dict
    @staticmethod
    def read_split(fold_path, split_name):
        """Read data from splitted txt"""
        split_eids = []
        with open(os.path.join(fold_path, f"{split_name}.txt"), encoding='utf-8') as reader:
            for line in reader:
                len_line = len(line.strip())
                if  len_line == 0:
                    continue
                eid = int(line.split('\t')[0])
                split_eids.append(eid)
        return split_eids
    @property
    def get_ds_name(self):
        """Get property of dataset name"""
        return self.ds_name
    @property
    def get_db_path(self):
        """Get property of database path"""
        return self.db_path
    