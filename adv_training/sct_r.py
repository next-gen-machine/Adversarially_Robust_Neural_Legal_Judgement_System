#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 00:23:17 2023

@author: rohit
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import textattack
import transformers


def train():
    #for legal BERT
    #model = transformers.AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased",num_labels = 100)
    #tokenizer = transformers.AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    
    #for legal RoBERTa
    model = transformers.AutoModelForSequenceClassification.from_pretrained("roberta-base",num_labels = 14)
    tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)


    #train dataset usng hugging face
    attack = textattack.attack_recipes.a2t_yoo_2021.A2TYoo2021.build(model_wrapper)
    #attack =  textattack.attack_recipes.textfooler_jin_2019.TextFoolerJin2019.build(model_wrapper)
    train_dataset = textattack.datasets.HuggingFaceDataset("lex_glue","scotus", split="train")
    eval_dataset = textattack.datasets.HuggingFaceDataset("lex_glue","scotus",split="test")
    
    
    #for  BERT
    #model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels = 14)
    #tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    
    
    #training arguments for text-attack
    training_args = textattack.TrainingArgs(

    num_epochs=3,
    num_clean_epochs=1,
    num_train_adv_examples=100,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    log_to_tb=True,)
    
    #trainer for text-attack
    trainer = textattack.Trainer(
        model_wrapper,
        "classification",
         attack,
        train_dataset,
        eval_dataset,
        training_args

        )
    trainer.train()



if __name__=="__main__":
    print('Running training script for ledgar')
    train()