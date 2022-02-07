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

class InputFeatures:
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id, ori_tokens, ori_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ori_tokens = ori_tokens
        self.ori_labels = ori_labels
class Utils:
    """As helpers"""
    def __init__(self):
        self.root_dir = os.getcwd()
    def is_uri(self, string):
        """To check, is the string URI?"""
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        try:
            url = re.findall(regex,string)
            result = [x[0] for x in url]
        except Exception:
            result = []
        found = False
        if len(result) > 0:
            found = True
        return found
    def get_label_of_entity(self, uri, endpoint):
        """Get entity label from knowledge base"""
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
        """Get label from URI resource"""
        word = str(ent)
        if '#' in ent:
            word = word.rsplit('#', maxsplit=1)[-1]
        else:
            last = word.rsplit('/', maxsplit=1)[-1]
            if last == '':
                num = len(word.split('/')) - 2
                last = word.split('/')[num]
            word = last
            if ':' in word:
                word = word.split(':')[-1]
        return word.title()
    def as_hours(self, seconds):
        """Convert second to hours, minutes, seconds"""
        minutes = math.floor(seconds / 60)
        hours = math.floor(minutes / 60)
        seconds -= minutes * 60
        minutes -= hours * 60
        return '%dh %dm %ds' % (hours, minutes, seconds)
    def read_epochs_from_log(self, ds_name, topk):
        """Read best epochs of the model"""
        log_file_path = os.path.join(self.root_dir, 'GATES_log.txt')
        key = f'{ds_name}-top{topk}'
        epoch_list = None
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(key):
                    epoch_list = list(eval(line.split('\t')[1]))
        return epoch_list
    def get_embeddings(self, word_emb_model):
        """Get embedding vectors"""
        if word_emb_model == "fasttext":
            word_emb = KeyedVectors.load_word2vec_format(os.path.join(self.root_dir, "data_inputs/kge/wiki-news-300d-1M.vec"))
        elif word_emb_model=="Glove":
            word_emb = {}
            with open("data_inputs/text_embed/glove.6B/glove.6B.300d.txt", 'r') as reader:
                for line in reader:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    word_emb[word] = vector
        else:
            print("please choose the correct word embedding model")
            sys.exit()
        return word_emb
    def mem(self):
        """Get current memory usage"""
        memory = psutil.cpu_percent()
        return memory
    def build_dict(self, f_path):
        """Build the vocabulary"""
        word2ix = {}
        with open(f_path, "r", encoding="utf-8") as reader:
            for _, pair in enumerate(reader):
                try:
                    temp = pair.strip().split("\t")
                    word2ix[temp[1]] = int(temp[0])
                except Exception:
                    print(temp)
        return word2ix
    def build_vec(self, word2ix, word_embedding):
        """Build vector representation"""
        word2vec = {}
        for word in word2ix:
            word2vec[word] = word_embedding[int(word2ix[word])]
        return word2vec
    def load_kg_embed(self, ds_name, emb_model):
        """Load pre-trained graph embeddings"""
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
    def tensor_from_data(self, pred_dict, facts, text_embed):
        """Get triple encoding"""
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
                except Exception:
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
        """Convert inputs to the features for BERT inputs"""
        features = []
        label_ids = self.tensor_from_weight(len(facts), facts, labels)
        for i, (_, pred_literal, obj_literal) in enumerate(t_literals):
            tokens_a = tokenizer.tokenize(pred_literal)
            tokens_b = tokenizer.tokenize(obj_literal)
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
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
    def counter(self, cur_dict, word):
        """Counter words"""
        if word in cur_dict:
            cur_dict[word] += 1
        else:
            cur_dict[word] = 1
    def tensor_from_weight(self, tensor_size, facts, label):
        """Convert label to weight tensor"""
        weight_tensor = torch.zeros(tensor_size)
        for label_word in label:
            order = -1
            for _, pred, obj in facts:
                order += 1
                data_word = f"{pred}++$++{obj}"
                if label_word == data_word:
                    weight_tensor[order] += label[label_word]
                    break
        return weight_tensor / torch.sum(weight_tensor)
    def normalize_adj(self, matrix):
        """Row-normalize sparse matrix"""
        rowsum = np.array(matrix.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return matrix.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    def normalize_features(self, matrix):
        """Row-normalize sparse matrix"""
        rowsum = np.array(matrix.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        matrix = r_mat_inv.dot(matrix)
        return matrix
    