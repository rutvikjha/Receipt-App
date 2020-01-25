# Receipt-App

The goal of this project is to create a machine learning algorithm reads in product names from a receipt and classifies into one of five categories: 1) Food/drink 2) Entertainment 3) Clothing 4) Home/Office , add 5) Misc after training.
(The goal is currently being worked on).

  Datasets containing items from each of these categories were assembled from ecommerce websites, amazon, homedepot, and jcpenny using beautiful soup (a webscraper). The datasets were preprocessed and split into a training set and a test set using RStudio (data_preprocessing_ecommerce.R). A tfidf vectorizer converted the training and test sets into word vectors (tfidf_frame.py and tfidf_test.py). The next step of the project is to implement the Naive Bayes algorithm to train a machine learning model to classify product names based on the tfidf word vectors.
  
training_set -> tfidf vectorizer -> Naive Bayes Classifier -> Machine Learning Model.


