    # -*- coding: utf-8 -*-
    """
    Created on Tue Jan  7 16:36:42 2020
    
    @author: rjsta
    """
    import gzip
    import gensim
    from gensim.models import Word2Vec 
    import logging
    import os
    import nltk
    
    import csv
    
    product_names = []

    with open('test_set.csv', encoding = "latin1") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
    
        for row in readCSV:
            product_names.append(row[0])
            #print(row)
    
    model2 = gensim.models.Word2Vec([product_names], min_count = 1, size = 100, 
                                             window = 5, sg = 0) 
    print(model2.wv.similarity("Grape", "Hammer"))
    
    
    
        
    #print(your_list)
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)