#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import gensim
import pickle
import random
import torch
from tqdm import tqdm
from tokenization import *
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from collections import Counter
import sys
import nltk
import string
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)


class DocDataset(Dataset):
    def __init__(self,
                 task_name,
                 texts,
                 labels,
                 txt_path=None,
                 no_below=5, no_above=0.1,
                 hasLable=False, rebuild=False, use_tfidf=False):
        cwd = os.getcwd()
        tmp_dir = os.path.join(cwd, 'data', task_name)
        if labels is None:
            self.labels = [None for i in range(len(texts))]
        else:
            self.labels = labels
        self.txtLines = texts
        self.dictionary = None
        self.bows, self.docs = None, None
        self.use_tfidf = use_tfidf
        self.tfidf, self.tfidf_model = None, None
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        if not rebuild and os.path.exists(os.path.join(tmp_dir, 'corpus.mm')):
            self.bows = gensim.corpora.MmCorpus(os.path.join(tmp_dir, 'corpus.mm'))
            if self.use_tfidf:
                self.tfidf = gensim.corpora.MmCorpus(os.path.join(tmp_dir, 'tfidf.mm'))
            self.dictionary = Dictionary.load_from_text(os.path.join(tmp_dir, 'dict.txt'))
            self.docs = pickle.load(open(os.path.join(tmp_dir, 'docs.pkl'), 'rb'))
            self.dictionary.id2token = {v: k for k, v in
                                        self.dictionary.token2id.items()}  # because id2token is empty be default, it is a bug.
        if rebuild:
            print('start downloading nltk words')
            nltk.download('words')
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger_eng')
            print('complete downloading nltk words')

            self.stopwords = nltk.corpus.stopwords.words("english")
            self.stemmer = nltk.stem.PorterStemmer()
            self.lemmatizer = nltk.stem.WordNetLemmatizer()

            # self.txtLines is the list of string, without any preprocessing.
            # self.texts is the list of list of tokens.

            print('Tokenizing ...')
            clean_texts = []
            clean_texts_tokenized = []
            for text in texts:
                clean_text, tokens = self.__pre_process(text)
                clean_texts.append(clean_text)
                clean_texts_tokenized.append(tokens)

            self.docs = clean_texts_tokenized
            # build dictionary
            self.dictionary = Dictionary(self.docs)
            # self.dictionary.filter_n_most_frequent(remove_n=20)
            self.dictionary.filter_extremes(no_below=no_below, no_above=no_above,
                                            keep_n=None)  # use Dictionary to remove un-relevant tokens
            self.dictionary.compactify()
            self.dictionary.id2token = {v: k for k, v in
                                        self.dictionary.token2id.items()}  # because id2token is empty by default, it is a bug.
            # convert to BOW representation
            self.bows, _docs = [], []
            for doc in self.docs:
                _bow = self.dictionary.doc2bow(doc)
                if _bow:
                    _docs.append(list(doc))
                    self.bows.append(_bow)
            self.docs = _docs
            if self.use_tfidf:
                self.tfidf_model = TfidfModel(self.bows)
                self.tfidf = [self.tfidf_model[bow] for bow in self.bows]
            # serialize the dictionary
            gensim.corpora.MmCorpus.serialize(os.path.join(tmp_dir, 'corpus.mm'), self.bows)
            self.dictionary.save_as_text(os.path.join(tmp_dir, 'dict.txt'))
            pickle.dump(self.docs, open(os.path.join(tmp_dir, 'docs.pkl'), 'wb'))
            if self.use_tfidf:
                gensim.corpora.MmCorpus.serialize(os.path.join(tmp_dir, 'tfidf.mm'), self.tfidf)
        self.vocab_size = len(self.dictionary)
        self.numDocs = len(self.bows)
        print(f'Processed {len(self.bows)} documents.')


    def __pre_process(self, text):

        # 1-Remove extra whitespace
        # remove leading and trailing white space
        text = text.strip()

        # replace multiple consecutive white space characters with a single space
        text = " ".join(text.split())

        # 2-Remove URLs

        # define a regular expression pattern to match URLs
        pattern = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"

        # replace URLs with an empty string
        text = re.sub(pattern, "", text)

        # Remove HTML code
        # define a regular expression pattern to match HTML tags
        pattern = r"<[^>]+>"

        # replace HTML tags with an empty string
        text = re.sub(pattern, "", text)

        # Remove digits
        text = re.sub(r'\b\w*\d\w*\b', '', text)  # Remove words with digits
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

        # 3-tokenize the text
        tokens = nltk.word_tokenize(text)

        # 4-lowercase the tokens
        tokens = [token.lower() for token in tokens]

        # 5-remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]

        # 6-remove stopwords
        # get list of stopwords in English

        tokens = [token for token in tokens if token.lower() not in self.stopwords]

        # get list of English words

        # words = nltk.corpus.words.words()

        # Spelling correction
        # correct spelling of each word
        # corrected_tokens = []
        # for token in tokens:
        #     # find the word with the lowest edit distance
        #     corrected_token = min(words, key=lambda x: nltk.edit_distance(x, token))
        #     corrected_tokens.append(corrected_token)
        #
        # tokens = corrected_tokens

        # Stemming
        # create stemmer object

        # stem each token
        tokens = [self.stemmer.stem(token) for token in tokens]

        # Lemmatization
        # create lemmatizer object

        # lemmatize each token
        # print(f'tokens: {tokens}')
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Part-of-speech tagging

        # tag the tokens with their POS tags
        tagged_tokens = nltk.pos_tag(tokens)

        return ' '.join(tokens), tokens

    def __getitem__(self, idx):
        bow = torch.zeros(self.vocab_size)
        if self.use_tfidf:
            item = list(zip(*self.tfidf[idx]))
        else:
            item = list(zip(*self.bows[idx]))  # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt, bow

    def __len__(self):
        return self.numDocs

    def collate_fn(self, batch_data):
        texts, bows = list(zip(*batch_data))
        return texts, torch.stack(bows, dim=0)

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def show_dfs_topk(self, topk=20):
        ndoc = len(self.docs)
        dfs_topk = sorted([(self.dictionary.id2token[k], fq) for k, fq in self.dictionary.dfs.items()],
                          key=lambda x: x[1], reverse=True)[:topk]
        for i, (word, freq) in enumerate(dfs_topk):
            print(f'{i + 1}:{word} --> {freq}/{ndoc} = {(1.0 * freq / ndoc):>.13f}')
        return dfs_topk

    def show_cfs_topk(self, topk=20):
        ntokens = sum([v for k, v in self.dictionary.cfs.items()])
        cfs_topk = sorted([(self.dictionary.id2token[k], fq) for k, fq in self.dictionary.cfs.items()],
                          key=lambda x: x[1], reverse=True)[:topk]
        for i, (word, freq) in enumerate(cfs_topk):
            print(f'{i + 1}:{word} --> {freq}/{ntokens} = {(1.0 * freq / ntokens):>.13f}')

    def topk_dfs(self, topk=20):
        ndoc = len(self.docs)
        dfs_topk = self.show_dfs_topk(topk=topk)
        return 1.0 * dfs_topk[-1][-1] / ndoc


'''
class DocDataLoader:
    def __init__(self,dataset=None,batch_size=128,shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idxes = list(range(len(dataset)))
        self.length = len(self.idxes)

    def __iter__(self):
        return self

    def __next__(self):
        if self.shuffle==True:
            random.shuffle(self.idxes)
        for i in range(0,self.length,self.batch_size):
            batch_ids = self.idxes[i:i+self.batch_size]
            batch_data = self.dataset[batch_ids]
            yield batch_data
'''


class TestData(Dataset):
    def __init__(self, dictionary=None, txtPath=None, lang="zh", tokenizer=None, stopwords=None, no_below=5,
                 no_above=0.1, use_tfidf=False):
        cwd = os.getcwd()
        self.txtLines = [line.strip('\n') for line in open(txtPath, 'r', encoding='utf-8')]
        self.dictionary = dictionary
        self.bows, self.docs = None, None
        self.use_tfidf = use_tfidf
        self.tfidf, self.tfidf_model = None, None

        print('Tokenizing ...')

        self.docs = tokenizer.tokenize(self.txtLines)
        # convert to BOW representation
        self.bows, _docs = [], []
        for doc in self.docs:
            if doc is not None:
                _bow = self.dictionary.doc2bow(doc)
                if _bow != []:
                    _docs.append(list(doc))
                    self.bows.append(_bow)
                else:
                    _docs.append(None)
                    self.bows.append(None)
            else:
                _docs.append(None)
                self.bows.append(None)
        self.docs = _docs
        if self.use_tfidf == True:
            self.tfidf_model = TfidfModel(self.bows)
            self.tfidf = [self.tfidf_model[bow] for bow in self.bows]
        self.vocabsize = len(self.dictionary)
        self.numDocs = len(self.bows)
        print(f'Processed {len(self.bows)} documents.')

    def __getitem__(self, idx):
        bow = torch.zeros(self.vocabsize)
        if self.use_tfidf:
            item = list(zip(*self.tfidf[idx]))
        else:
            item = list(zip(*self.bows[idx]))  # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt, bow

    def __len__(self):
        return self.numDocs

    def __iter__(self):
        for doc in self.docs:
            yield doc

