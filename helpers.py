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

from SPARQLWrapper import SPARQLWrapper, JSON
from gensim.models.keyedvectors import KeyedVectors

nltk.download('punkt')

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
    
    # Build dictionary word to index
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
    
    # Build word to vector
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
        #print("#####")
        #print(edesc)
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
            #print("tokens", tokens, obj)
            for token in tokens:
                try:
                    vec = text_embed[token]
                except:
                    vec = np.zeros([300,])
                    #flag=False
                #print(token, vec, vec.shape)
                arrays.append(vec)
            if len(tokens)>1:    
                obj_vector = np.average(arrays, axis=0)
            else:
                obj_vector = arrays[0]
            #print(obj)
            #print(obj_vector)
            arrays_obj_literal_list.append(obj_vector)
        #arrays_obj_literal_list = np.array(arrays_obj_literal_list)
        #print(arrays_obj_literal_list.shape)
        obj_tensor = torch.tensor(arrays_obj_literal_list).unsqueeze(1)
      
        return pred_tensor, obj_tensor