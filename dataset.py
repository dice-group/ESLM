#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:58:42 2021

@author: asep
"""
import os
from nltk.corpus import stopwords
from rdflib.plugins.parsers.ntriples import NTriplesParser, Sink

from helpers import Utils
from config import config
from stop_words import get_stop_words


utils = Utils()

class ESBenchmark(object):
    """This class contains modules pertaining to dataset processes"""
    def __init__(self, file_n = 6, topk=5, weighted_adjacency_matrix=False):
        self.file_n = file_n
        self.topk = topk
        self.weighted_adjacency_matrix = weighted_adjacency_matrix
        self.value_error = "The database's name must be dbpedia, lmdb. or faces"
        self.in_esbm_dir = os.path.join(os.getcwd(), "datasets/ESBM_benchmark_v1.2")
        self.in_faces_dir = os.path.join(os.getcwd(), 'datasets/FACES/faces_data')
    def get_5fold_train_valid_test_elist(self, ds_name_str):
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
        split_eids = []
        with open(os.path.join(fold_path, "{}.txt".format(split_name)),encoding='utf-8') as reader:
            for line in reader:
                if len(line.strip())==0:
                    continue
                eid = int(line.split('\t')[0])
                split_eids.append(eid)
        return split_eids
    def get_triples(self, ds_name, num):
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
            def triple(self,s,p,o):
                sub = s.toPython()
                pred = p.toPython()
                obj = o.toPython()
                triple_tuple = (sub, pred, obj)
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
        for sub, pred, obj in triples:
            if utils.is_URI(obj):
                obj_literal = utils.get_label_of_entity(obj, endpoint)
            else:
                obj_literal = obj
            pred_literal = utils.get_label_of_entity(pred, endpoint)
            sub_literal = utils.get_label_of_entity(sub, endpoint)
            triple = (sub_literal, pred_literal, obj_literal)
            triples_tuple.append(triple)
        return triples_tuple
    def get_literals(self, ds_name, num):
        triples_literal=[]
        with open(os.path.join(os.getcwd(), "data_inputs/literals/{}".format(ds_name), "{}_literal.txt".format(num))) as reader:
            for literal in reader:
                values = literal.split("\t")
                sub_literal = values[0]
                pred_literal = values[1]
                obj_literal = values[2].replace("\n", "")
                triple_literal_tuple = (sub_literal, pred_literal, obj_literal)
                triples_literal.append(triple_literal_tuple)
        return triples_literal
    def get_training_dataset(self, ds_name):
        train_eids, valid_eids, _ = self.get_5fold_train_valid_test_elist(ds_name)
        train_data = []
        valid_data = []
        for fold, eids_per_fold in enumerate(train_eids):
            train_data_perfold = []
            edesc = {}
            for eid in eids_per_fold:
                triples = self.get_triples(ds_name, eid)
                edesc[eid] = triples
            train_data_perfold.append(edesc)
            train_data.append(train_data_perfold)
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
        for fold, eids_per_fold in enumerate(test_eids):
            test_data_perfold = list()
            for eid in eids_per_fold:
                triples = self.get_triples(ds_name, eid)
                edesc = {}
                edesc[eid] = triples
                test_data_perfold.append(edesc)
            test_data.append(test_data_perfold)
        return test_data
    def prepare_labels(self, ds_name, num, top_n, file_n):
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
            i = 0
            j = 0
            def triple(self,sub,pred,obj):
                #parse s,p,o to dictionaries/databases
                sub = sub.toPython()
                pred = pred.toPython()
                obj = obj.toPython()
                triple_tuple = (sub, pred, obj)
                triples.append(triple_tuple)
        IndexSink = IndexSink()      
        for i in range(file_n):
            triples = []
            parser = NTriplesParser(IndexSink)
            with open(os.path.join(db_path, "{}".format(num), "{}_gold_top{}_{}.nt".format(num, top_n, i)), 'rb') as reader:
                parser.parse(reader)
            for s, p, o in triples:
                utils.counter(per_entity_label_dict, "{}++$++{}".format(p, o))
        return per_entity_label_dict
    def get_predicates_corpus(self, ds_name):
        train_data, _ = self.get_training_dataset(ds_name)
        utils = Utils()
        pred_dict_all = {}
        for fold, data in enumerate(train_data):
            pred_dict = {}
            for t_fold in data:
                for eid in t_fold.keys():
                    for triple in t_fold[eid]:
                        _, pred, _ = triple
                        pred = utils.extract_triple_to_word(pred.lower())
                        if pred not in pred_dict:
                            pred_dict[pred]=len(pred_dict)        
            pred_dict_all[fold] = pred_dict                
        return pred_dict_all
    def get_sub_obj_corpus(self, ds_name):
        train_data, _ = self.get_training_dataset(ds_name)
        utils = Utils()
        d_sub = {}
        d_obj = {}
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
            d_sub[fold] = d_sub_fold
            d_obj[fold] = d_obj_fold
        return d_sub, d_obj
    def get_stop_words(self):
        stop_words = list(get_stop_words('en'))         #About 900 stopwords
        nltk_words = list(stopwords.words('english')) #About 150 stopwords
        stop_words.extend(nltk_words)
        return stop_words
    def get_gold_summaries(self, ds_name, num, topk):
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
            i = 0
            j = 0
            def triple(self,sub,pred,obj):
                sub = sub.toPython()
                pred = pred.toPython()
                obj = obj.toPython()
                triple_tuple = (sub, pred, obj)
                triples_summary.append(triple_tuple)
        gold_summary_list = []
        IndexSink = IndexSink()
        parser = NTriplesParser(IndexSink)
        for i in range(config["file_n"]):
            with open(os.path.join(db_path, "{}".format(num), "{}_gold_top{}_{}.nt".format(num, topk, i)), 'rb') as reader:
                parser.parse(reader)
            n_list = []
            for triple in triples_summary:
                gold_id = triples_dict[triple]
                n_list.append(gold_id)
            gold_summary_list.append(n_list)
        return gold_summary_list
    