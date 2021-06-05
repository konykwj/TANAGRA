#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:23:31 2019

@author: bill
"""
import pickle
import re
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models.phrases import Phrases, Phraser

def get_wordnet_pos(treebank_tag):
    #Converts fromt the part of speech output from the get_wordnet_pos function and the actual part of speech.
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def text_clean(doc, wnl, stopword_list):
    #Cleans up the text by removing all tabs and newlines, lemmatizes, removing special characters and digits, and stopwords.
    doc = re.sub(r"\t|\n", " ", doc) #Remove all tabs and newlines
    
    tokens = word_tokenize(doc) #Tokenizes for lemmatization
    pos_tokens = pos_tag(tokens) #Part of speech tagging. At this point the text is pdocerved to enable good tagging.
    doc_list = []
    for i in range(len(pos_tokens)):
        if len(get_wordnet_pos(pos_tokens[i][1]))>0:
            try:
                word = pos_tokens[i][0]
                word = re.sub(r"\W", "", word) #Removes special characters
                word = re.sub(r'\d','',word) #Removes digits
                word = wnl.lemmatize(word, pos = get_wordnet_pos(pos_tokens[i][1])).lower()
                if len(word) > 1 and word not in stopword_list:
                    doc_list.append(word)        
            except:
                print(pos_tokens[i])
    
    doc = re.sub(r' {2,}',' ', ' '.join(doc_list))
        
    return doc

#Process for cleaning:
def define_stopwords(custom_stopwords):
    
    from nltk.corpus import stopwords
    
    stopwords_list = stopwords.words('english')
    
    for word in custom_stopwords:
        stopwords_list.append(word)
    print('Stopwords defined')
    print(custom_stopwords)
    
    return stopwords_list

#stopwords = stopwords.words('english')
def create_corpus(text_df, stopword_list, text_key):
    
    wnl = WordNetLemmatizer()
    
    corpus = []
    for doc in text_df[text_key].values:
        corpus.append(text_clean(doc, wnl, stopword_list))
        
    print('Corpus created')
        
    return corpus

def identify_bigrams(corpus, gensim_sentences, gensim_min_count, gensim_threshold):
        
    bigram = Phraser(Phrases(gensim_sentences, min_count = gensim_min_count, threshold = gensim_threshold))
        
    for i in range(len(corpus)):
        phrase_sentence = bigram[gensim_sentences[i]]
        
        new_sentence =  [x for x in phrase_sentence]
             
        gensim_sentences[i] = new_sentence
        corpus[i] = ' '.join(new_sentence)
    
    print('Bigrams identified')
    
    return corpus, gensim_sentences

def create_gensim_sentences(corpus):
    return [word_tokenize(doc) for doc in corpus]


def create_w2v_skipgram_model(gensim_sentences, w2v_location, min_word_count):
   
    w2v_model = gensim.models.Word2Vec(gensim_sentences, min_count=min_word_count)

    w2v_model.save(w2v_location)
    
    print('Skipgram word-2-vec model created')
    
def vectorize_corpus(corpus, vectorized_topic_location, min_word_count):

    tfidf_vect = TfidfVectorizer(min_df=min_word_count) # Creates the vectorizer instance
    
    word_tfidf = tfidf_vect.fit_transform(corpus) #vectorizes the corpus
    tfidf_word_list = tfidf_vect.get_feature_names() #Gets the feature names
    pickle.dump([word_tfidf,tfidf_word_list], open( vectorized_topic_location, "wb" ) ) #Saves the file.
    
    print('Corpus Vectorized')

