#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import time
import datetime
import numpy as np
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from rich.console import Console
from distutils.util import strtobool

from evaluator.map import MAP
from evaluator.fmeasure import FMeasure
from evaluator.ndcg import NDCG
from config import config
from classes.helpers import Utils
from classes.dataset import ESBenchmark

UTILS = Utils()
LOSS_FUNCTION = config["loss_function"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_model='bert-base-uncased'
TOKENIZER = AutoTokenizer.from_pretrained(pretrained_model)
# define a rich console logger
console=Console(record=True)
    
class BertClassifier(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', nb_class=1, kg_hidden=1200):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert.modules())[-2].out_features
        self.softmax = nn.Softmax(dim=0)
        for param in self.bert.parameters():
            param.requires_grad = True
        if 'hidden_size' in self.bert.config.to_dict():
            embedding_dim = self.bert.config.to_dict()['hidden_size']
        else:
            embedding_dim = self.bert.config.to_dict()['dim']

        self.out = nn.Linear(embedding_dim*2, 1)

        self.fc1 = nn.Linear(kg_hidden, kg_hidden)
        nn.init.normal_(self.fc1.weight,mean=0,std=0.01)
        torch.nn.init.constant_(self.fc1.bias, 0)
        self.fc2 = nn.Linear(kg_hidden,embedding_dim)
        nn.init.normal_(self.fc2.weight,mean=0,std=0.01)
        torch.nn.init.constant_(self.fc2.bias, 0)
        self.ReLU = nn.ReLU()
        seq = [self.fc1, self.ReLU, self.fc2]
        self.seq = nn.Sequential(*seq)
        self.softmax = nn.Softmax(dim=0)
    def forward(self, input_ids, attention_mask, token_type_ids, kg_embedding):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) 
        #self.bert_model(input_ids, attention_mask)[0][:, 0]
        seq_out = outputs.last_hidden_state
        batch_size, num_token, _ = seq_out.shape
        kg_out = self.seq(kg_embedding)
        cat_out = torch.cat((seq_out, seq_out*kg_out/((seq_out.shape[-1])**0.5)), dim=2)
        cat_out = self.out(cat_out)
        cat_out = cat_out[:, -1, :]
        cls_logit = self.softmax(cat_out)
        return cls_logit

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def build_dictionary(input_dict):
    f = open(input_dict, "r")
    content = f.readlines()
    idx2dict = dict()
    for items in content:
        items = items.replace("\n", "")
        items = items.split("\t")
        idx = int(items[0])
        dict_value = items[1]
        idx2dict[dict_value]=idx
    f.close()
    return idx2dict

def load_dglke(ds_name):
    """Load pre-trained graph embeddings"""
    input_entity_dict = f"{ds_name}-esbm/entities.tsv"
    input_relation_dict = f"{ds_name}-esbm/relations.tsv" 
    entity2ix = build_dictionary(input_entity_dict)
    pred2ix = build_dictionary(input_relation_dict)
    entity2vec = np.load(f'{ds_name}-kge-model/ComplEx_{ds_name}/{ds_name}_ComplEx_entity.npy', mmap_mode='r')
    pred2vec = np.load(f'{ds_name}-kge-model/ComplEx_{ds_name}/{ds_name}_ComplEx_relation.npy', mmap_mode='r')
    return entity2vec, pred2vec, entity2ix, pred2ix
def main(mode, best_epoch):
    """Main module"""
    file_n = config["file_n"]
    is_weighted_adjacency_matrix = config["weighted_adjacency_matrix"]
    for ds_name in config["ds_name"]:
        if ds_name == "dbpedia":
            MAX_LENGTH = 46
        elif ds_name == "faces":
            MAX_LENGTH = 46
        else:
            MAX_LENGTH = 34

        ent_emb_dim = 400
        pred_emb_dim = 400

        entity2vec, pred2vec, entity2ix, pred2ix = load_dglke(ds_name)
        entity_dict = entity2vec
        pred_dict = pred2vec
        pred2ix_size = len(pred2ix)
        entity2ix_size = len(entity2ix)
        hidden_size = ent_emb_dim + pred_emb_dim 

        if mode == "test":
            for topk in config["topk"]:
                dataset = ESBenchmark(ds_name, file_n, topk, is_weighted_adjacency_matrix)
                test_data = dataset.get_testing_dataset()
                models = []
                for fold in range(5):
                    models_path = os.path.join("models", f"bert_dglke_checkpoint-{ds_name}-{topk}-{fold}")
                    model = BertClassifier()
                    if bool(strtobool(best_epoch)) is True:
                        checkpoint = torch.load(os.path.join(models_path, f"checkpoint_best_{fold}.pt"))
                    else:
                        checkpoint = torch.load(os.path.join(models_path, f"checkpoint_latest_{fold}.pt"))
                    model.load_state_dict(checkpoint["model"])
                    model.to(DEVICE)
                    model.eval()
                    models.append(model)
                    
                for fold in range(5):
                    print("")
                    print(f"fold: {fold+1}, total entities: {len(test_data[fold][0])}", f"topk: top{topk}")
                    generated_entity_summaries(test_data[fold][0], dataset, topk, fold, models, MAX_LENGTH, entity_dict, pred_dict, entity2ix, pred2ix)
                evaluation(dataset, topk)

def generated_entity_summaries(test_data, dataset, topk, fold, models, max_length, entity_dict, pred_dict, entity2ix, pred2ix):
    """"Generated entity summaries"""
    with torch.no_grad():
        for eid in test_data:
            triples = dataset.get_triples(eid)
            literal = dataset.get_literals(eid)
            labels = dataset.prepare_labels(eid)

            p_embs, o_embs, s_embs = [], [], []
            for triple in triples:
                s, p, o = triple
                o = str(o)
                o_emb = np.zeros([400,])
                if o.startswith("http://"):
                    oidx = entity2ix[o]
                    try:
                        o_emb = entity_dict[oidx]
                    except:
                        pass
                p_emb = np.zeros([400,])
                if p in pred2ix:
                    pidx=pred2ix[p]
                    try:
                        p_emb = pred_dict[pidx]
                    except:
                        pass
                s_emb = np.zeros([400,])
                if s in entity2ix:
                    sidx=entity2ix[s]
                    try:
                        s_emb = entity_dict[sidx]
                    except:
                        pass
                s_embs.append(s_emb)
                o_embs.append(o_emb)
                p_embs.append(p_emb)
            s_tensor = torch.tensor(np.array(s_embs),dtype=torch.float).unsqueeze(1)
            o_tensor = torch.tensor(np.array(o_embs),dtype=torch.float).unsqueeze(1)
            p_tensor = torch.tensor(np.array(p_embs),dtype=torch.float).unsqueeze(1)
            #kg_embeds = torch.cat((p_tensor, o_tensor), 2).to(DEVICE)
            #kg_embeds = torch.add(p_tensor, o_tensor).to(DEVICE)
            kg_embeds = torch.cat((s_tensor, p_tensor, o_tensor), 2).to(DEVICE)

            features = UTILS.convert_to_features_with_subject(literal, TOKENIZER, max_length, triples, labels)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(DEVICE)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(DEVICE)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(DEVICE)
            target_tensor = UTILS.tensor_from_weight(len(triples), triples, labels).to(DEVICE)
            output_tensor = evaluate_n_members(models, fold, all_input_ids, all_input_mask, all_segment_ids, kg_embeds)
            output_tensor = output_tensor.view(1, -1).cpu()
            target_tensor = target_tensor.view(1, -1).cpu()
            #(label_top_scores, label_top) = torch.topk(target_tensor, topk)
            _, output_top = torch.topk(output_tensor, topk)
            _, output_rank = torch.topk(output_tensor, len(test_data[eid]))
            directory = f"outputs/{dataset.get_ds_name}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = f"outputs/{dataset.get_ds_name}/{eid}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            top_or_rank = "top"
            rank_list = output_top.squeeze(0).numpy().tolist()
            writer(dataset.get_db_path, directory, eid, top_or_rank, topk, rank_list)
            top_or_rank = "rank_top"
            rank_list = output_rank.squeeze(0).numpy().tolist()
            writer(dataset.get_db_path, directory, eid, top_or_rank, topk, rank_list)

def writer(db_dir, directory, eid, top_or_rank, topk, rank_list):
    "Write triples to file"
    with open(os.path.join(db_dir, f"{eid}", f"{eid}_desc.nt"), encoding="utf8") as fin:
        with open(os.path.join(directory, f"{eid}_{top_or_rank}{topk}.nt"), "w", encoding="utf8") as fout:
            triples = [triple for _, triple in enumerate(fin)]
            for rank in rank_list:
                fout.write(triples[rank])
# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, fold, all_input_ids, all_input_mask, all_segment_ids, kg_embeds):
    if fold==4:
        subset = [members[0],  members[4]]
    else:
        subset = [members[fold],  members[fold+1]]
    yhat = ensemble_predictions(subset, all_input_ids, all_input_mask, all_segment_ids, kg_embeds)
    return yhat

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, all_input_ids, all_input_mask, all_segment_ids, kg_embeds):
	# make predictions
    yhats = torch.stack([model(all_input_ids, all_input_mask, all_segment_ids, kg_embeds) for model in members])
    result = torch.sum(yhats, axis=0)
    return result

def get_rank_triples(db_path, num, top_n, triples_dict):
  triples=[]
  encoded_triples = []
  filename = os.path.join(db_path, "{}".format(num), "{}_rank.nt".format(num))
  if os.path.exists(os.path.join(db_path, "{}".format(num), "{}_rank_top{}.nt".format(num, top_n))):
      filename = os.path.join(db_path, "{}".format(num), "{}_rank_top{}.nt".format(num, top_n))
  with open(filename, encoding="utf8") as reader:   
    for i, triple in enumerate(reader):
        triple = triple.replace("\n", "").strip()
        triples.append(triple)
        
        encoded_triple = triples_dict[triple]
        encoded_triples.append(encoded_triple)
  return triples, encoded_triples

def get_topk_triples(db_path, num, top_n, triples_dict):
  triples=[]
  encoded_triples = []
  
  with open(os.path.join(db_path, "{}".format(num), "{}_top{}.nt".format(num, top_n)), encoding="utf8") as reader:   
    for i, triple in enumerate(reader):
        triple = triple.replace("\n", "").strip()
        triples.append(triple)
        
        encoded_triple = triples_dict[triple]
        encoded_triples.append(encoded_triple)
  return triples, encoded_triples


def get_all_data(db_path, num, top_n, file_n):
  import glob
  triples_dict = {}
  triple_tuples = []
  ### Retrieve all triples of an entity based on eid
  with open(os.path.join(db_path, "{}".format(num), "{}_desc.nt".format(num)), encoding="utf8") as reader:   
    for i, triple in enumerate(reader):
      if len(triple)==1:
        continue  
      triple_tuple = triple.replace("\n", "").strip()#parserline(triple)
      triple_tuples.append(triple_tuple)
      if triple_tuple not in triples_dict:
        triples_dict[triple_tuple] = len(triples_dict)
  gold_list = []
  ds_name = db_path.split("/")[-1].split("_")[0]
  
  ### Get file_n/ n files of ground truth summaries for faces dataset
  if ds_name=="faces":
      gold_files = glob.glob(os.path.join(db_path, "{}".format(num), "{}_gold_top{}_*".format(num, top_n).format(num)))
      #print(len(gold_files))
      if len(gold_files) != file_n:
          file_n = len(gold_files)
  
  ### Retrieve ground truth summaries of an entity based on eid and total of file_n  
  for i in range(file_n):
    with open(os.path.join(db_path, 
            "{}".format(num), 
            "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)),
            encoding="utf8") as reader:
      #print(path.join(db_path, "{}".format(num), "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)))
      n_list = []
      for i, triple in enumerate(reader):
        if len(triple)==1:
            continue
        triple_tuple = triple.replace("\n", "").strip()#parserline(triple)
        gold_id = triples_dict[triple_tuple]
        n_list.append(gold_id)
      gold_list.append(n_list)
        
  return gold_list, triples_dict, triple_tuples

def evaluation(dataset, k):
    ndcg_class = NDCG()
    fmeasure = FMeasure()
    m = MAP()
    
    if dataset.ds_name == "dbpedia":
        IN_SUMM = os.path.join(os.getcwd(), 'outputs/dbpedia')
        start = [0, 140]
        end   = [100, 165]
    elif dataset.ds_name == "lmdb":
        IN_SUMM = os.path.join(os.getcwd(), 'outputs/lmdb')
        start = [100, 165]
        end   = [140, 175]
    elif dataset.ds_name == "faces":
        IN_SUMM = os.path.join(os.getcwd(), 'outputs/faces')
        start = [0, 25]
        end   = [25, 50]
            
    all_ndcg_scores = []
    all_fscore = []
    all_map_scores = []
    total_ndcg=0
    total_fscore=0
    total_map_score=0
    for i in range(start[0], end[0]):
        t = i+1
        gold_list_top, triples_dict, triple_tuples = get_all_data(dataset.db_path, t, k, dataset.file_n)
        rank_triples, encoded_rank_triples = get_rank_triples(IN_SUMM, t, k, triples_dict)
        topk_triples, encoded_topk_triples = get_topk_triples(IN_SUMM, t, k, triples_dict)
        #print("############### Top-K Triples ################", t)
        #print("######################")
        #print(triples_dict)
        #print("total of gold summaries", len(gold_list_top))
        #print("topk", encoded_topk_triples)
        #ndcg_score = getNDCG(rel)
        ndcg_score = ndcg_class.get_score(gold_list_top, encoded_rank_triples)
        f_score = fmeasure.get_score(encoded_topk_triples, gold_list_top)
        map_score = m.get_map(encoded_rank_triples, gold_list_top)
        #print(ndcg_score)
        #print("*************************")
        total_ndcg += ndcg_score
        all_ndcg_scores.append(ndcg_score)
        
        total_fscore += f_score
        all_fscore.append(f_score)
        
        all_map_scores.append(map_score)
        
    for i in range(start[1], end[1]):
        t = i+1
        gold_list_top, triples_dict, triple_tuples = get_all_data(dataset.db_path, t, k, dataset.file_n)
        rank_triples, encoded_rank_triples = get_rank_triples(IN_SUMM, t, k, triples_dict)
        topk_triples, encoded_topk_triples = get_topk_triples(IN_SUMM, t, k, triples_dict)
        #print("############### Top-K Triples ################", t)
        #print("######################")
        #print(triples_dict)
        #print("total of gold summaries", len(gold_list_top))
        #print("topk", encoded_topk_triples)
        #ndcg_score = getNDCG(rel)
        ndcg_score = ndcg_class.get_score(gold_list_top, encoded_rank_triples)
        f_score = fmeasure.get_score(encoded_topk_triples, gold_list_top)
        map_score = m.get_map(encoded_rank_triples, gold_list_top)
        #print(ndcg_score)
        #print("*************************")
        total_ndcg += ndcg_score
        all_ndcg_scores.append(ndcg_score)
        
        total_fscore += f_score
        all_fscore.append(f_score)
        
        all_map_scores.append(map_score)
        
    print("{}@top{}: F-Measure={}, NDCG={}, MAP={}".format(dataset, k, np.average(all_fscore), np.average(all_ndcg_scores), np.average(all_map_scores)))

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='BERT-GATES')
    PARSER.add_argument("--mode", type=str, default="test", help="mode type: train/test/all")
    PARSER.add_argument("--best_epoch", type=str, default="False", help="")
    ARGS = PARSER.parse_args()
    main(ARGS.mode, ARGS.best_epoch)
    
