import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch import optim
import numpy as np

from transformers import (
    AutoTokenizer, 
    T5Tokenizer,
    T5Model,
    get_linear_schedule_with_warmup
)


from classes.config import Config
from classes.helpers import Utils
from classes.data import (
    load_dglke, 
    format_triples, 
    format_time,
    writer
)
from classes.dataset import ESBenchmark
from classes.models import ESLMKGE, ESLM
from evaluation import evaluation

def main(args):
    config = Config(args)
    do_train = config.do_train
    do_test = config.do_test
    device = config.device
    model_name = config.model_name
    if model_name == "bert":
        model_base = "bert-base-uncased"
    elif model_name == "ernie":
        model_base = "nghuyong/ernie-2.0-en"
    elif model_name == "t5":
        model_base = "t5-base"
    else:
        print("please choose the correct model name: bert/ernie/t5")
        sys.exit()
    if config.enrichment:
        main_model_dir = f"models-eslm-kge-{model_name}"
    else:
        main_model_dir = f"models-eslm-{model_name}"
    criterion = nn.BCELoss()#BCELoss()  # Assuming a regression task
    utils = Utils()
    if model_name=="t5":
        tokenizer = T5Tokenizer.from_pretrained(f'{model_base}', model_max_length=config.max_length, legacy=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"{model_base}", model_max_length=config.max_length)
    
    # Training part
    if do_train == True:
        print("Training on progress ....")
        for ds_name in config.ds_name:
            print(f"Dataset: {ds_name}")
            if config.enrichment:
                entity2vec, pred2vec, entity2ix, pred2ix = load_dglke(ds_name)
                entity_dict = entity2vec
                pred_dict = pred2vec
                pred2ix_size = len(pred2ix)
                entity2ix_size = len(entity2ix)
            for topk in config.topk:
                dataset = ESBenchmark(ds_name, 6, topk, False)
                train_data, valid_data = dataset.get_training_dataset()
                for fold in range(config.k_fold):
                    train_data_size = len(train_data[fold][0])
                    train_data_samples = train_data[fold][0]
                    print(f"fold: {fold+1}, total entities: {train_data_size}", f"topk: top{topk}")
                    models_path = os.path.join(f"{main_model_dir}", f"eslm_checkpoint-{ds_name}-{topk}-{fold}")
                    models_dir = os.path.join(os.getcwd(), models_path)
                    if not os.path.exists(models_dir):
                        os.makedirs(models_dir)
                    if config.enrichment:
                        model = ESLMKGE(model_name, model_base)
                    else:
                        model = ESLM(model_name, model_base)
                    param_optimizer = list(model.named_parameters())
                    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
                    optimizer_parameters = [
                        {
                            "params": [
                                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": 0.001,
                        },
                        {
                            "params": [
                                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": 0.0,
                        },
                    ]
                    optimizer = optim.AdamW(optimizer_parameters, lr=config.learning_rate)
                    num_training_steps = train_data_size * config.epochs
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=0,
                        num_training_steps=num_training_steps
                    )
                    for epoch in range(config.epochs):
                        model.train()
                        model.to(config.device)
                        #Training part
                        t_start = time.time()
                        train_loss = 0
                        for num, eid in enumerate(train_data_samples):
                            triples = dataset.get_triples(eid)
                            labels = dataset.prepare_labels(eid)
                            literals = dataset.get_literals(eid)
                            triples_formatted = format_triples(literals)
                            input_ids_list = []
                            attention_masks_list = []
                            for triple in triples_formatted:
                                src_tokenized = tokenizer.encode_plus(
                                    triple, 
                                    max_length=config.max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_attention_mask=True,
                                    return_token_type_ids=True,
                                    add_special_tokens=True
                                    #return_tensors='pt'
                                )
                                src_input_ids = src_tokenized['input_ids']
                                src_attention_mask = src_tokenized['attention_mask']
                                src_segment_ids = src_tokenized['token_type_ids']
                                input_ids_list.append(src_input_ids)
                                attention_masks_list.append(src_attention_mask)

                            ### apply kge
                            if config.enrichment:
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
                                kg_embeds = torch.cat((s_tensor, p_tensor, o_tensor), 2).to(device)
                                ### end apply kge

                            input_ids_tensor = torch.tensor(input_ids_list).to(device)
                            attention_masks_tensor = torch.tensor(attention_masks_list).to(device)
                            targets = utils.tensor_from_weight(len(triples), triples, labels).to(device)
                            if config.enrichment:
                                outputs = model(input_ids_tensor, attention_masks_tensor, kg_embeds)
                            else:
                                outputs = model(input_ids_tensor, attention_masks_tensor)

                            # Reshaping the logits
                            reshaped_logits = outputs
                            #print(reshaped_logits)
                            # Ensure your targets tensor is of shape [103, 1]
                            reshaped_targets = targets.unsqueeze(-1)
                            # Now compute the loss
                            loss = criterion(reshaped_logits, reshaped_targets)
                            optimizer.zero_grad()
                            loss.backward()

                            # Gradient clipping (optional)
                            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                            optimizer.step()
                            scheduler.step()

                            train_loss += loss.item()
                        avg_train_loss = train_loss/train_data_size
                        training_time = format_time(time.time() - t_start)
                        # Evaluation part
                        t_start = time.time()
                        valid_data_size = len(valid_data[fold][0])
                        valid_data_samples = valid_data[fold][0]
                        model.eval()
                        with torch.no_grad():
                            valid_loss = 0
                            valid_acc = 0
                            for eid in valid_data_samples:
                                triples = dataset.get_triples(eid)
                                labels = dataset.prepare_labels(eid)
                                literals = dataset.get_literals(eid)
                                triples_formatted = format_triples(literals)
                                input_ids_list = []
                                attention_masks_list = []
                                for triple in triples_formatted:
                                    src_tokenized = tokenizer.encode_plus(
                                        triple, 
                                        max_length=config.max_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_attention_mask=True,
                                        add_special_tokens=True
                                        #return_tensors='pt'
                                    )
                                    src_input_ids = src_tokenized['input_ids']
                                    src_attention_mask = src_tokenized['attention_mask']
                                    input_ids_list.append(src_input_ids)
                                    attention_masks_list.append(src_attention_mask)

                                ### apply kge
                                if config.enrichment:
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
                                    kg_embeds = torch.cat((s_tensor, p_tensor, o_tensor), 2).to(device)
                                ### end apply kge

                                input_ids_tensor = torch.tensor(input_ids_list).to(device)
                                attention_masks_tensor = torch.tensor(attention_masks_list).to(device)
                                targets = utils.tensor_from_weight(len(triples), triples, labels).to(device)
                                if config.enrichment:
                                    outputs = model(input_ids_tensor, attention_masks_tensor, kg_embeds)
                                else:
                                    outputs = model(input_ids_tensor, attention_masks_tensor)
                                # Reshaping the logits
                                reshaped_logits = outputs
                                # Ensure your targets tensor is of shape [103, 1]
                                reshaped_targets = targets.unsqueeze(-1)
                                # Now compute the loss
                                loss = criterion(reshaped_logits, reshaped_targets)
                                valid_loss += loss.item()
                                valid_output_tensor = reshaped_logits.view(1, -1).cpu()
                                (_, output_top) = torch.topk(valid_output_tensor, topk)
                                triples_dict = dataset.triples_dictionary(eid)
                                gold_list_top = dataset.get_gold_summaries(eid, triples_dict)
                                acc = utils.accuracy(output_top.squeeze(0).numpy().tolist(), gold_list_top)
                                valid_acc += acc
                            avg_valid_loss = valid_loss/valid_data_size
                            avg_valid_acc = valid_acc/valid_data_size
                            validation_time = format_time(time.time() - t_start)
                            torch.save({
                                "epoch": epoch,
                                "model": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "train_loss": avg_train_loss,
                                'valid_loss': avg_valid_loss,
                                'fold': fold,
                                'training_time': training_time,
                                'validation_time': validation_time
                                }, os.path.join(models_dir, f"checkpoint_latest_{fold}.pt"))
                            print(f"epoch:{epoch}, train-loss:{avg_train_loss}")
        print("Training model is completed.")

    # Testing part
    if do_test == True:
        print("Predicting on progress ....")
        for ds_name in config.ds_name:
            if config.enrichment:
                entity2vec, pred2vec, entity2ix, pred2ix = load_dglke(ds_name)
                entity_dict = entity2vec
                pred_dict = pred2vec
                pred2ix_size = len(pred2ix)
                entity2ix_size = len(entity2ix)
            for topk in config.topk:
                dataset = ESBenchmark(ds_name, 6, topk, False)
                test_data = dataset.get_testing_dataset()
                for fold in range(config.k_fold):
                    test_data_size = len(test_data[fold][0])
                    test_data_samples = test_data[fold][0]
                    if config.enrichment:
                        model = ESLMKGE(model_name, model_base)
                    else:
                        model = ESLM(model_name, model_base)
                    models_path = os.path.join(f"{main_model_dir}", f"eslm_checkpoint-{ds_name}-{topk}-{fold}")
                    try:
                        checkpoint = torch.load(os.path.join(models_path, f"checkpoint_latest_{fold}.pt"))
                    except:
                        print("Error while loading the model")
                        sys.exit()
                    model.load_state_dict(checkpoint["model"])
                    model.eval()
                    model.to(device)
                    with torch.no_grad():
                        for eid in test_data_samples:
                            triples = dataset.get_triples(eid)
                            labels = dataset.prepare_labels(eid)
                            literals = dataset.get_literals(eid)
                            triples_formatted = format_triples(literals)
                            input_ids_list = []
                            attention_masks_list = []
                            for triple in triples_formatted:
                                src_tokenized = tokenizer.encode_plus(
                                    triple, 
                                    max_length=config.max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_attention_mask=True,
                                    add_special_tokens=True
                                )
                                src_input_ids = src_tokenized['input_ids']
                                src_attention_mask = src_tokenized['attention_mask']
                                input_ids_list.append(src_input_ids)
                                attention_masks_list.append(src_attention_mask)

                            ### apply kge
                            if config.enrichment:
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
                                kg_embeds = torch.cat((s_tensor, p_tensor, o_tensor), 2).to(device)
                            ### end apply kge
                            input_ids_tensor = torch.tensor(input_ids_list).to(device)
                            attention_masks_tensor = torch.tensor(attention_masks_list).to(device)
                            batch_size = input_ids_tensor.size(0)
                            targets = utils.tensor_from_weight(len(triples), triples, labels).to(device)
                            if config.enrichment:
                                outputs = model(input_ids_tensor, attention_masks_tensor, kg_embeds)
                            else:
                                outputs = model(input_ids_tensor, attention_masks_tensor)
                            # Reshaping the logits
                            reshaped_logits = outputs
                            # Ensure your targets tensor is of shape [103, 1]
                            reshaped_targets = targets.unsqueeze(-1)
                            reshaped_logits = reshaped_logits.view(1, -1).cpu()
                            reshaped_targets = reshaped_targets.view(1, -1).cpu()
                            _, output_top = torch.topk(reshaped_logits, topk)
                            _, output_rank = torch.topk(reshaped_logits, len(test_data_samples[eid]))
                            directory = f"outputs-{model_name}/{dataset.get_ds_name}"
                            if not os.path.exists(directory):
                                os.makedirs(directory)
                            directory = f"outputs-{model_name}/{dataset.get_ds_name}/{eid}"
                            if not os.path.exists(directory):
                                os.makedirs(directory)
                            top_or_rank = "top"
                            rank_list = output_top.squeeze(0).numpy().tolist()
                            writer(dataset.get_db_path, directory, eid, top_or_rank, topk, rank_list)
                            top_or_rank = "rank_top"
                            rank_list = output_rank.squeeze(0).numpy().tolist()
                            writer(dataset.get_db_path, directory, eid, top_or_rank, topk, rank_list)
        print("Predicting is completed")
        
        print("Evaluation on progress ...")
        for ds_name in config.ds_name:
            for topk in config.topk:
                dataset = ESBenchmark(ds_name, 6, topk, False)
                print(ds_name)
                evaluation(dataset, topk, model_name)
        print("Evaluation is done ...")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ESLM')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.set_defaults(train=True)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no-test', dest='test', action='store_false')
    parser.set_defaults(test=True)
    parser.add_argument('--enrichment', action='store_true')
    parser.add_argument('--no-enrichment', dest='enrichment', action='store_false')
    parser.set_defaults(enrichment=True)
    parser.add_argument("--model", type=str, default="", help="")
    parser.add_argument("--max_length", type=int, default=40, help="")
    parser.add_argument("--epochs", type=int, default=10, help="")
    parser.add_argument("--learning_rate", type=int, default=5e-5, help="")
    args = parser.parse_args()
    main(args)