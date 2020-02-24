# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:50:55 2020

@author: cucusenok

"""

import nltk
from nltk import sent_tokenize, word_tokenize, regexp_tokenize
import pymorphy2




class RussianWordTokenaizer:

    def __init__(self, text, stopWords = None, normalize = True):
        
        self.stopWords = stopWords
        self._text = text
        self.normalize = normalize
    
    def make(self):
        return self.tokenizeAndLemmatize()
    
    
    def tokenizeAndLemmatize(self, regexp=r'(?u)\b\w{4,}\b'):
        words = [w for sent in sent_tokenize(self._text)
                 for w in regexp_tokenize(sent, regexp)]
        if self.normalize:
            words = self.tokenNormalize(words)
        if self.stopWords:
            words = self.removeStopwWords(words)
        return words
    
    
    def tokenNormalize(self, tokens):
        morph = pymorphy2.MorphAnalyzer()
        return [morph.parse(tok)[0].normal_form for tok in tokens]


    def removeStopwWords(self, tokens, min_length=4):
        if not self.stopWords:
            return tokens
        
        self.stopWords = set(self.stopWords)

        tokens = [tok
                  for tok in tokens
                  if tok not in self.stopWords and (int(len(tok)) >= min_length)]
        return tokens