# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:10:54 2020

@author: rjsta
"""

#Assigns tfidf scores to words in product_names
#Try to Fix: the words of the first entry seem to be given more priority tfidf scores just because they appear first

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import csv

#preprocess data (code originally from w2v.py)
product_names = []

with open('test_set.csv', encoding = "latin1") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    
    for row in readCSV:
        product_names.append(row[0])
    
del product_names[0] #deletes ttle of column


#CALCULATING TFIDF SCORES USING TfidfVectorizer (short method)
#parameters for count vectorizer are stated here
tfidf_vectorizer = TfidfVectorizer(use_idf = True)

#input data set here
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(product_names)

#pull first vector of first element in docs
first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]

# place tf-idf values in a pandas data frame
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)