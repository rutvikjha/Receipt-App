# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:31:17 2020

@author: rjsta
"""

#https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.XhZllUdKhPY

#tfidf score classifies a group based on its unique atributes (words)
#and downplays its common words because they are likely to appear in other groups as well

#could also be useful in keyword analyzer

import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

docs=["the house had a tiny little mouse",
      "the cat saw the mouse",
      "the mouse ran away from the house",
      "the cat finally ate the mouse",
      "the end of the mouse story"
     ]
 
#CALCULATING TFIDF SCORES USING TfidfTransformer (long method)
#creates instance of count vectorizer
    #CountVectorizer can be further utlized to preprocess data
    #Use a custom list of stop words, maximum/minimum word size
cv = CountVectorizer()

#counts words in list
word_count_vector = cv.fit_transform(docs)
#word_count_vector.shape() returns (5,16); 5 elements in docs and 16 unique words

#Create IDF values (Calculate how important words are to list docs)
tfidf_transformer = TfidfTransformer(smooth_idf = True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

#Visualize the IDF values using dataframes
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
 
# sort ascending // the lower IDF value a word has the less unique it to the document (the word is quite frequent)
df_idf.sort_values(by=['idf_weights'])

#COMPUTING TFIDF SCORES // tfidf is a score where word frequency is weighted/scaled by its uniqueness
#count matrix // computes the word counts of docs and stores it in matrix form
count_vector = cv.transform(docs)

#computer TFIDF scores
tf_idf_vector = tfidf_transformer.transform(count_vector)

#displaying tfidf values in dataframe
feature_names = cv.get_feature_names()
#get tfidf vector for first document
first_document_vector = tf_idf_vector[0]

#print tfidf scores
#print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
 

from sklearn.feature_extraction.text import TfidfVectorizer 
#CALCULATING TFIDF SCORES USING TfidfVectorizer (short method)
#parameters for count vectorizer are stated here
tfidf_vectorizer = TfidfVectorizer(use_idf = True)

#input data set here
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs)

#pull first vector of first element in docs
first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]

# place tf-idf values in a pandas data frame
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)



 
# another alternative way to calculate tfidf values by calling fit and transform seperately
tfidf_vectorizer=TfidfVectorizer(use_idf=True)

fitted_vectorizer=tfidf_vectorizer.fit(docs)
tfidf_vectorizer_vectors=fitted_vectorizer.transform(docs)