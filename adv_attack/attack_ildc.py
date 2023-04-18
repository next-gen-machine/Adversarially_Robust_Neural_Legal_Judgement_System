#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 19:33:04 2023

@author: rohit
"""


import textattack
import transformers
import pandas as pd


def train():
    #model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels = 2)
    #tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    model = transformers.AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased",num_labels = 2)
    tokenizer = transformers.AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    #model = transformers.AutoModelForSequenceClassification.from_pretrained("roberta-base",num_labels = 2)
    #tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    
    attack = textattack.attack_recipes.a2t_yoo_2021.A2TYoo2021.build(model_wrapper)
    
    
    #dataset
    train_data = pd.read_csv("/home/rohit/adv_robust_legal_paper/english/train_510_ildc.csv")
    eval_data = pd.read_csv("/home/rohit/adv_robust_legal_paper/english/valid_510_ildc.csv")
    
    #train_data = train_data[:500]
    #eval_data = eval_data[:500]
    
    #data preprocessing
    data = []
    for i in range(train_data.shape[0]):
        data.append((train_data.iloc[i][0],int(train_data.iloc[i][1])))
        
    data_valid = []
    for i in range(eval_data.shape[0]):
        data_valid.append((eval_data.iloc[i][0],int(eval_data.iloc[i][1])))
        
        
        
    #ildc_train = textattack.datasets.Dataset(data)
    ildc_valid = textattack.datasets.Dataset(data_valid)
    
    
    #Attack 20 samples with CSV logging and checkpoint saved every 5 interval
    attack_args = textattack.AttackArgs(
        num_examples=20,
        log_to_csv="log.csv",
        checkpoint_interval=5,
        checkpoint_dir="checkpoints",
        disable_stdout=True)
    
    
    attacker = textattack.Attacker(attack, ildc_valid, attack_args)
    attacker.attack_dataset()
    
        
    
        
    
        
    
        
    
   
    
    
        
    
    
    
    
    
if __name__=="__main__":
    print('Running training script for ildc')
    train()
