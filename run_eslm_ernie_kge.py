#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import time
import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as f
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from rich.console import Console
from distutils.util import strtobool
from rdflib.plugins.parsers.ntriples import NTriplesParser, Sink

from evaluator.map import MAP
from evaluator.fmeasure import FMeasure
from evaluator.ndcg import NDCG
from config import config
from classes.helpers import Utils
from classes.dataset import ESBenchmark

UTILS = Utils()
LOSS_FUNCTION = config["loss_function"]
DEVICE =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_model='nghuyong/ernie-2.0-base-en'
TOKENIZER = AutoTokenizer.from_pretrained(pretrained_model)
console=Console(record=True)
    
class ErnieClassifier(nn.Module):
    def __init__(self, pretrained_model='nghuyong/ernie-2.0-base-en', nb_class=1, kg_hidden=1200):
        super(ErnieClassifier, self).__init__()
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
    #entity2vec = np.load(f'{ds_name}-kge-model/DistMult_{ds_name}/{ds_name}_DistMult_entity.npy', mmap_mode='r')
    #pred2vec = np.load(f'{ds_name}-kge-model/DistMult_{ds_name}/{ds_name}_DistMult_relation.npy', mmap_mode='r')
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
        
        if mode == "train":
            for topk in config["topk"]:
                dataset = ESBenchmark(ds_name, file_n, topk, is_weighted_adjacency_matrix)
                train_data, valid_data = dataset.get_training_dataset()
                for fold in range(5):
                    fold = fold
                    print("")
                    print(f"Fold: {fold+1}, total entities: {len(train_data[fold][0])}", f"topk: top{topk}")
                    model = ErnieClassifier()
                    model.to(DEVICE)
                    param_optimizer = list(model.named_parameters())
                    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                    optimizer_grouped_parameters = [
                        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                        ]
                    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config["learning_rate"], eps=1e-8)
                    models_path = os.path.join("models", f"ernie_dglke_checkpoint-{ds_name}-{topk}-{fold}")
                    models_dir = os.path.join(os.getcwd(), models_path)
                    train(model, optimizer, train_data[fold][0], valid_data[fold][0], dataset, topk, fold, models_dir, MAX_LENGTH, entity_dict, pred_dict, entity2ix, pred2ix)
        elif mode == "test":
            for topk in config["topk"]:
                dataset = ESBenchmark(ds_name, file_n, topk, is_weighted_adjacency_matrix)
                test_data = dataset.get_testing_dataset()
                for fold in range(5):
                    print("")
                    print(f"fold: {fold+1}, total entities: {len(test_data[fold][0])}", f"topk: top{topk}")
                    models_path = os.path.join("models", f"ernie_dglke_checkpoint-{ds_name}-{topk}-{fold}")
                    model = ErnieClassifier()
                    if bool(strtobool(best_epoch)) is True:
                        checkpoint = torch.load(os.path.join(models_path, f"checkpoint_best_{fold}.pt"))
                    else:
                        checkpoint = torch.load(os.path.join(models_path, f"checkpoint_latest_{fold}.pt"))
                    model.load_state_dict(checkpoint["model"])
                    model.to(DEVICE)
                    generated_entity_summaries(model, test_data[fold][0], dataset, topk, MAX_LENGTH, entity_dict, pred_dict, entity2ix, pred2ix)
                evaluation(dataset, topk)
def train(model, optimizer, train_data, valid_data, dataset, topk, fold, models_dir, max_length, entity_dict, pred_dict, entity2ix, pred2ix):
    """Training module"""
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    best_acc = 0
    stop_valid_epoch = None
    total_steps = len(train_data) * config["n_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    for epoch in range(config["n_epochs"]):
        model.train()
        train_loss = 0
        train_acc = 0
        print("")
        print(f'======== Epoch {epoch+1} / {config["n_epochs"]} ========')
        print('Training...')
        t_start = time.time()
        for eid in tqdm(train_data):
            triples = dataset.get_triples(eid)
            labels = dataset.prepare_labels(eid)
            literal = dataset.get_literals(eid)

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
            kg_embeds = torch.cat((s_tensor, p_tensor, o_tensor), 2).to(DEVICE)
            
            features = UTILS.convert_to_features_with_subject(literal, TOKENIZER, max_length, triples, labels)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(DEVICE)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(DEVICE)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(DEVICE)
            target_tensor = UTILS.tensor_from_weight(len(triples), triples, labels).to(DEVICE)
            output_tensor = model(all_input_ids, all_input_mask, all_segment_ids, kg_embeds)
            loss = LOSS_FUNCTION(output_tensor.view(-1), target_tensor.view(-1)).to(DEVICE)
            train_output_tensor = output_tensor.view(1, -1).cpu()
            (_, output_top) = torch.topk(train_output_tensor, topk)
            triples_dict = dataset.triples_dictionary(eid)
            gold_list_top = dataset.get_gold_summaries(eid, triples_dict)
            acc = UTILS.accuracy(output_top.squeeze(0).numpy().tolist(), gold_list_top)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            train_acc += acc
        training_time = format_time(time.time() - t_start)
        print("  Training epcoh took: {:}".format(training_time))
        valid_acc = 0
        valid_loss = 0
        print("")
        print("Running Validation...")
        t_start = time.time()
        model.eval()
        with torch.no_grad():
            for eid in tqdm(valid_data):
                triples = dataset.get_triples(eid)
                labels = dataset.prepare_labels(eid)
                literal = dataset.get_literals(eid)

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
                kg_embeds = torch.cat((s_tensor, p_tensor, o_tensor), 2).to(DEVICE)

                features = UTILS.convert_to_features_with_subject(literal, TOKENIZER, max_length, triples, labels)
                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(DEVICE)
                all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(DEVICE)
                all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(DEVICE)
                target_tensor = UTILS.tensor_from_weight(len(triples), triples, labels).to(DEVICE)
                output_tensor = model(all_input_ids, all_input_mask, all_segment_ids, kg_embeds)
                loss = LOSS_FUNCTION(output_tensor.view(-1), target_tensor.view(-1)).to(DEVICE)
                valid_output_tensor = output_tensor.view(1, -1).cpu()
                (_, output_top) = torch.topk(valid_output_tensor, topk)
                triples_dict = dataset.triples_dictionary(eid)
                gold_list_top = dataset.get_gold_summaries(eid, triples_dict)
                acc = UTILS.accuracy(output_top.squeeze(0).numpy().tolist(), gold_list_top)
                valid_loss += loss.item()
                valid_acc += acc
        validation_time = format_time(time.time() - t_start)
        print("  Validation took: {:}".format(validation_time))
        train_loss = train_loss/len(train_data)
        train_acc = train_acc/len(train_data)
        valid_loss = valid_loss/len(valid_data)
        valid_acc = valid_acc/len(valid_data)
        curr_lr = optimizer.param_groups[0]['lr']
        print("")
        print(f"train-loss:{train_loss}, train-acc:{train_acc}, valid-loss:{valid_loss}, valid-acc:{valid_acc}, current lr: {curr_lr}")
        torch.save({
                "epoch": epoch,
                "bert_model": model.bert.state_dict(),
                "model":model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                'valid_loss': valid_loss,
                'fold': fold,
                'acc': best_acc,
                'training_time': training_time,
                'validation_time': validation_time
                }, os.path.join(models_dir, f"checkpoint_latest_{fold}.pt"))
       
    return stop_valid_epoch
def generated_entity_summaries(model, test_data, dataset, topk, max_length, entity_dict, pred_dict, entity2ix, pred2ix):
    """"Generated entity summaries"""
    model.eval()
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
            output_tensor = model(all_input_ids, all_input_mask, all_segment_ids, kg_embeds)
            output_tensor = output_tensor.view(1, -1).cpu()
            target_tensor = target_tensor.view(1, -1).cpu()
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
        ndcg_score = ndcg_class.get_score(gold_list_top, encoded_rank_triples)
        f_score = fmeasure.get_score(encoded_topk_triples, gold_list_top)
        map_score = m.get_map(encoded_rank_triples, gold_list_top)
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
        ndcg_score = ndcg_class.get_score(gold_list_top, encoded_rank_triples)
        f_score = fmeasure.get_score(encoded_topk_triples, gold_list_top)
        map_score = m.get_map(encoded_rank_triples, gold_list_top)
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
 
