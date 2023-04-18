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
    model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels = 14)
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    # model = transformers.AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased",num_labels = 14)
    # tokenizer = transformers.AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    #model = transformers.AutoModelForSequenceClassification.from_pretrained("roberta-base",num_labels = 14)
    #tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    
    attack = textattack.attack_recipes.a2t_yoo_2021.A2TYoo2021.build(model_wrapper)
    
    
    dataset = textattack.datasets.HuggingFaceDataset("lex_glue","scotus", split="test")
    
    #data preprocessing
    
        
        
        
    
    
    
    
    #Attack 20 samples with CSV logging and checkpoint saved every 5 interval
    attack_args = textattack.AttackArgs(
        num_examples=1000,
        log_to_csv="log.csv",
        checkpoint_interval=5,
        checkpoint_dir="checkpoints",
        disable_stdout=True)
    
    
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()
    
        
    
        
    
        
    
        
    
   
    
    
        
    
    
    
    
    
if __name__=="__main__":
    print('Running training script for scotus')
    train()
