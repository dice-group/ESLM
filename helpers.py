#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:14:07 2021

@author: asep
"""
import re
import os
import math
import numpy as np
import sys
import psutil
import torch
import nltk
import scipy.sparse as sp

from SPARQLWrapper import SPARQLWrapper, JSON
from gensim.models.keyedvectors import KeyedVectors

nltk.download('punkt')

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id, ori_tokens, ori_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ori_tokens = ori_tokens
        self.ori_labels = ori_labels
        
class Utils(object):
    def __init__(self):
        self.root_dir = os.getcwd()    
    def is_URI(self, string):
        # findall() has been used 
        # with valid conditions for urls in string
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        try:
            url = re.findall(regex,string)      
            result = [x[0] for x in url]
        except:
            result = []
        found = False
        if len(result) > 0:
          found = True        
        return found    
    def get_label_of_entity(self, uri, endpoint):
        sparql = SPARQLWrapper(endpoint)
        sparql.setQuery("""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?label
            WHERE { <%s> rdfs:label ?label }
        """ % (uri))
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()        
        for result in results["results"]["bindings"]:
            try:
                if result["label"]["xml:lang"] == "en":
                    words = result["label"]["value"]
                    return words.title()
            except:
                words = result["label"]["value"]
                return words        
        word = self.get_uri_label(uri)        
        return word    
    def get_uri_label(self, ent):
        word = str(ent)
        if '#' in ent:
            word = word.split('#')[-1]
        else:
            last = word.split('/')[-1]
            if last == '':
               num = len(word.split('/')) - 2
               last = word.split('/')[num]
            word = last
            if ':' in word:
                word = word.split(':')[-1]
        return word.title()    
    def asHours(self, s):
    	m = math.floor(s / 60)
    	h = math.floor(m / 60)
    	s -= m * 60
    	m -= h * 60
    	return '%dh %dm %ds' % (h, m, s)    
    def read_epochs_from_log(self, ds_name, topk):
        log_file_path = os.path.join(self.root_dir, 'GATES_log.txt')
        
        key = '{}-top{}'.format(ds_name, topk)
        epoch_list = None
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(key):
                    epoch_list = list(eval(line.split('\t')[1]))
        return epoch_list
    def get_embeddings(self, word_emb_model):
        if word_emb_model == "fasttext":
            word_emb = KeyedVectors.load_word2vec_format(os.path.join(self.root_dir, "data_inputs/kge/wiki-news-300d-1M.vec"))
        elif word_emb_model=="Glove":
            word_emb = {}
            with open("data_inputs/text_embed/glove.6B/glove.6B.300d.txt", 'r') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    word_emb[word] = vector
        else:
            print("please choose the correct word embedding model")
            sys.exit()
        return word_emb
    def mem(self):
    	memory = psutil.cpu_percent()
    	return memory
    def build_dict(self, f_path):
        word2ix = {}
        with open(f_path, "r", encoding="utf-8") as f:
            for _, pair in enumerate(f):
                try:
                    temp = pair.strip().split("\t")
                    word2ix[temp[1]] = int(temp[0])
                except:
                    print(temp)
        return word2ix
    def build_vec(self, word2ix, word_embedding):
        word2vec = {}
        for word in word2ix:
            word2vec[word] = word_embedding[int(word2ix[word])]
        return word2vec
    def load_kg_embed(self, ds_name, emb_model):
        directory = os.path.join(self.root_dir, "data_inputs/kg_embed/{}/".format(ds_name))
        entity2ix = self.build_dict(os.path.join(directory, "entities.dict"))
        pred2ix = self.build_dict(os.path.join(directory, "relations.dict"))
        if emb_model =="DistMult":
           embedding = np.load(os.path.join(directory, "DistMult_vec.npz"))
        elif emb_model == "ComplEx":
           embedding = np.load(os.path.join(directory, "ComplEx_vec.npz"))
        elif emb_model == "ConEx":
           embedding = np.load(os.path.join(directory, "ConEx_vec.npz"))   
        else:
           raise ValueError("Please choose KGE DistMult or ComplEx")
        entity_embedding = embedding["ent_embedding"]
        pred_embedding = embedding["rel_embedding"]
        entity2vec = self.build_vec(entity2ix, entity_embedding)
        pred2vec = self.build_vec(pred2ix, pred_embedding)
        return entity2vec, pred2vec, entity2ix, pred2ix
    def tensor_from_data(self, entity_dict, pred_dict, facts, literals, text_embed):
        '''
        Tensor concatenation model 1 is obtained by the concatenation of KGE (DistMult/ComplEx) as predicate embeddings and
        word embeddings as object embeddings 
        '''
        pred_list, obj_list, obj_literal_list = [], [], []
        for _, _, pred, obj, obj_literal in facts:
            pred_list.append(pred_dict[pred])
            obj_list.append(obj)
            obj_literal_list.append(obj_literal)
        pred_tensor = torch.tensor(pred_list).unsqueeze(1)
        arrays_obj_literal_list=[]
        for obj in obj_literal_list:
            arrays=[]
            tokens = nltk.word_tokenize(obj)
            for token in tokens:
                try:
                    vec = text_embed[token]
                except:
                    vec = np.zeros([300,])
                arrays.append(vec)
            if len(tokens)>1:    
                obj_vector = np.average(arrays, axis=0)
            else:
                obj_vector = arrays[0]
            arrays_obj_literal_list.append(obj_vector)
        obj_tensor = torch.tensor(arrays_obj_literal_list).unsqueeze(1)
        return pred_tensor, obj_tensor
    def convert_to_features(self, t_literals, tokenizer, max_sequence_length, facts, labels):
        features = []
        label_ids = self.tensor_from_weight(len(facts), facts, labels)
        for i, (sl, pl, ol) in enumerate(t_literals):
            tokens_a = tokenizer.tokenize(pl)
            tokens_b = tokenizer.tokenize(ol)
            self.truncate_seq_pair(tokens_a, tokens_b, max_sequence_length - 3)
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_sequence_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            label_id = label_ids[i]
            assert len(input_ids) == max_sequence_length
            assert len(input_mask) == max_sequence_length
            assert len(segment_ids) == max_sequence_length
            features.append(InputFeatures(
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              ori_tokens=tokens,
                              ori_labels=labels 
                            ))
        return features
    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()        
    def counter(self, cur_dict, word):
        if word in cur_dict:
            cur_dict[word] += 1
        else:
            cur_dict[word] = 1
    # Define label/target tensor
    def tensor_from_weight(self, tensor_size, facts, label):
        weight_tensor = torch.zeros(tensor_size)
        for label_word in label:
            order = -1
            for _, pred, obj in facts:
                order += 1
                data_word = "{}++$++{}".format(pred, obj)
                if label_word == data_word:
                    weight_tensor[order] += label[label_word]
                    break
        return weight_tensor / torch.sum(weight_tensor)
    # adapted from pygat
    def normalize_adj(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    # adapted from pygat
    def normalize_features(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    