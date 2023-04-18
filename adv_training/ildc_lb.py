#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 00:30:25 2023

@author: rohit
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import textattack
import transformers
import pandas as pd



def train():
    #set path according
    train_data = pd.read_csv("/data2/home/rohitr/ar_nlj/datasets/ildc/train_510_ildc.csv")
    eval_data = pd.read_csv("/data2/home/rohitr/ar_nlj/datasets/ildc/valid_510_ildc.csv")
    
    data = []
    for i in range(train_data.shape[0]):
        data.append((train_data.iloc[i][0],int(train_data.iloc[i][1])))
        
        
    data_valid = []
    for i in range(eval_data.shape[0]):
        data_valid.append((eval_data.iloc[i][0],int(eval_data.iloc[i][1])))
        
        
    ildc_train = textattack.datasets.Dataset(data)
    ildc_valid = textattack.datasets.Dataset(data_valid)
    
    
    #model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels = 2)
    #tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    
    
    #for legal BERT
    model = transformers.AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased",num_labels = 2)
    tokenizer = transformers.AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    
    #for RoBERTa
    #model = transformers.AutoModelForSequenceClassification.from_pretrained("roberta-base",num_labels = 2)
    #tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)


    #train dataset using hugging face
    attack = textattack.attack_recipes.a2t_yoo_2021.A2TYoo2021.build(model_wrapper)
    #attack =  textattack.attack_recipes.textfooler_jin_2019.TextFoolerJin2019.build(model_wrapper)
    
    
    
    #training arguments for text-attack
    training_args = textattack.TrainingArgs(

    num_epochs=3,
    num_clean_epochs=1,
    num_train_adv_examples=1000,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    )
    
    #trainer for text-attack
    trainer = textattack.Trainer(
        model_wrapper,
        "classification",
         attack,
        ildc_train,
        ildc_valid,
        training_args

        )
    trainer.train()
    
    
if __name__=="__main__":
    print('Running training script for ildc')
    train()