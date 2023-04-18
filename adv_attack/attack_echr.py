#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 19:33:04 2023

@author: rohit
"""


import textattack
import transformers
import pandas as pd
import re
#import nltk
from nltk.corpus import stopwords

def train():
    model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels = 14)
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    # model = transformers.AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased",num_labels = 14)
    # tokenizer = transformers.AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    #model = transformers.AutoModelForSequenceClassification.from_pretrained("roberta-base",num_labels = 14)
    #tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    
    attack = textattack.attack_recipes.a2t_yoo_2021.A2TYoo2021.build(model_wrapper)
    
    
    #dataset
    echr_a_train = pd.read_csv("/home/rohit/adv_robust_legal_paper/english/echr_/echr_a.csv")
    echr_a_test = pd.read_csv("/home/rohit/adv_robust_legal_paper/english/echr_a_test.csv")
    
    #data preprocessing
    
    def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
        # Clean the text, with the option to remove stopwords and to stem words.
        
        # Convert words to lower case and split them
        text = text.lower().split()
    
        # Optionally, remove stop words
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
        
        text = " ".join(text)
    
        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"[\([{})\]]", "", text)
        #text = re.sub(r"]"," ",text)
        
        # Optionally, shorten words to their stems
        #if stem_words:
        #    text = text.split()
        #    stemmer = SnowballStemmer('english')
        #    stemmed_words = [stemmer.stem(word) for word in text]
        #    text = " ".join(stemmed_words)
        
        # Return a list of words
        return(text)
    train_text = echr_a_train['text'].values
    test_text = echr_a_test['text'].values
    

    train_text = [text_to_wordlist(x) for x in train_text]
    test_text = [text_to_wordlist(x) for x in test_text]    
    
    
    train_label  = echr_a_train['label'].values
    test_label = echr_a_test['label'].values
    
    data = []
    for i in range(len(train_text)):
        data.append((train_text[i],int(train_label[i])))
        
    data_valid = []
    for i in range(len(test_text)):
        data_valid.append((test_text[i],int(test_label[i])))
        
        
    echr_a_train_ = textattack.datasets.Dataset(data)
    echr_a_test_ = textattack.datasets.Dataset(data_valid)
            
        
        
    
    
    
    
    #Attack 1000 samples with CSV logging and checkpoint saved every 5 interval
    attack_args = textattack.AttackArgs(
        num_examples=1000,
        log_to_csv="log.csv",
        checkpoint_interval=5,
        checkpoint_dir="checkpoints",
        disable_stdout=True)
    
    
    attacker = textattack.Attacker(attack, echr_a_test_, attack_args)
    attacker.attack_dataset()
    
        
    
        
    
        
    
        
    
   
    
    
        
    
    
    
    
    
if __name__=="__main__":
    print('Running training script for scotus')
    train()
