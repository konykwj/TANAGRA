#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:23:31 2019

@author: Bill Konyk
"""
import pickle
import re
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from gensim.models.phrases import Phrases, Phraser
import numpy as np
from scipy.sparse import spdiags
from sklearn.preprocessing import normalize


def get_wordnet_pos(treebank_tag):
    """
    Converts fromt the part of speech output from the NLTK's pos_tag function to the actual part of speech
    
    Parameters
    ----------
    treebank_tag : str
        Part of speech (POS) output from NLTK's pos_tag function'
        
    Returns
    -------
    pos : str
        Part of speech of the word as defined by NLTK's wordnet
    """
    
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
    """
    Cleans up the text by: 
        -lemmatizing words 
        -removing stop words
        -removing special characters
        -removing digits
    
    Parameters
    ----------
    doc : str
        Text to clean
    wnl: class
        Instance of NLTK's WordNetLemmatizer
    stopword_list: list
        List of common words (stop words) to ignore
        
    Returns
    -------
    doc : str
        Cleaned text
    """
    
    #Cleans up the text by removing all tabs and newlines, lemmatizes, removing special characters and digits, and stopwords.
    doc = re.sub(r"\t|\n", " ", doc) #Remove all tabs and newlines
    
    tokens = word_tokenize(doc) #Tokenizes for lemmatization
    pos_tokens = pos_tag(tokens) #Part of speech tagging. At this point the text is preserved to enable good tagging.
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


def define_stopwords(custom_stopwords, language = 'english'):
    """
    Creates a stopword list
    
    Parameters
    ----------
    custom_stopwords: list
        List of custom words to ignore
    language: string
        Language to pull stop words from NLTK. Default is english.
        
    Returns
    -------
    stopword_list : list
        List of all words that will be removed in processing
    """
    
    stopwords_list = stopwords.words(language) #Assume that the user is speaking english...
    
    for word in custom_stopwords:
        stopwords_list.append(word)
    
    return stopwords_list


def create_corpus(text_df, stopword_list, text_key):
    """
    Builds the corpus from a dataframe
    
    Parameters
    ----------
    text_df: pandas DataFrame
        Dataframe containing text information
    stopword_list: list
        List of stopwords to remove
    text_key: str
        Column in text_df containing the text
        
    Returns
    -------
    corpus : list
        List containing the cleaned text
    """
    
    wnl = WordNetLemmatizer()
    
    corpus = []
    for doc in text_df[text_key].values:
        corpus.append(text_clean(doc, wnl, stopword_list))
        
    return corpus

def identify_bigrams(corpus, gensim_min_count, gensim_threshold, gensim_sentences = []):
    """
    Identifies bigrams in the text
    
    Parameters
    ----------
    corpus : list
        List containing the cleaned text
    gensim_min_count: int
        Minimum number of times that words must appear together
    gensim_threshold: int
        Threshold for gensim's processing
        
    Returns
    -------
    corpus : list
        List containing the cleaned text including bigrams
    gensim_sentences : list
        A tokenized version of the corpus for use in creating the word-2-vec model
    """
    
    if len(gensim_sentences) == 0:
        gensim_sentences = create_gensim_sentences(corpus)
        
    bigram = Phraser(Phrases(gensim_sentences, min_count = gensim_min_count, threshold = gensim_threshold))
        
    for i in range(len(corpus)):
        phrase_sentence = bigram[gensim_sentences[i]]
        
        new_sentence =  [x for x in phrase_sentence]
             
        gensim_sentences[i] = new_sentence
        corpus[i] = ' '.join(new_sentence)
    
    print('Bigrams identified')
    
    return corpus, gensim_sentences

def create_gensim_sentences(corpus):
    """
    Tokenizes the corpus
    
    Parameters
    ----------
    corpus : list
        List containing the cleaned text
        
    Returns
    -------
    gensim_sentences : list
        A tokenized version of the corpus for use in creating the word-2-vec model
    """
    
    return [word_tokenize(doc) for doc in corpus]


def create_w2v_skipgram_model(gensim_sentences, w2v_location, min_word_count):
    """
    Uses gensim to create a Word2Vec model, where each word is converted into a vector using machine-learning techniques.
    
    Parameters
    ----------
    gensim_sentences : list
        A tokenized version of the corpus
    w2v_location: string
        File and path for where to save the model
    min_word_count: int
        Minimum numbe of times that words must appear
        
    Returns
    -------
    None
    """
   
    w2v_model = gensim.models.Word2Vec(gensim_sentences, min_count=min_word_count)

    w2v_model.save(w2v_location)
        
def vectorize_corpus(corpus, vectorized_topic_location, min_word_count, 
                     vec_type = 'tfidf',
                     log_base = 2):
    """
    Vectorize the corpus using the TF-IDF scheme
    
    Parameters
    ----------
    corpus : list
        List containing the cleaned text
    vectorized_topic_location: string
        File and path for where to save the vectorized model
    min_word_count: int
        Minimum numbe of times that words must appear to be counted
    vec_type: str
        Type of vectorization to use. Options are "info" or "tfidf"
    log_base: int
        Base of log scaling to use. Defaults to 2
  
        
    Returns
    -------
    None
    """
    
    
    if vec_type == 'info':
        
        cv = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df = min_word_count)

        doc_term_mat = cv.fit_transform(corpus)
                
        #Converting to binary; whether a word appears in the corpus or not
        count_mat = doc_term_mat.copy()
        count_mat[count_mat>0] = 1

        #Creates the number of times that each word occurrs in a document with another word
        co_occurrence_mat = count_mat.T.dot(count_mat)

        #Converting to the probability of finding a particular pair of words in a document
        co_occurrence_mat.data =  co_occurrence_mat.data / co_occurrence_mat.sum() 

        #The diagonal represents the probability of finding a given word in a document
        p_word = co_occurrence_mat.diagonal()
        p_word = p_word / p_word.sum()

        #Creating matrix for mutual information calculation
        # p_mat_inv = csc_array(np.diag(1/p_word))
        
        p_mat_inv = spdiags(1/p_word, 0, p_word.size, p_word.size)
        p_mat = p_mat_inv.copy()
        p_mat.data = 1/p_mat.data

        #converting the co_occurrence_mat into a joint probability distribution and calculating informaiton with it
        co_occurrence_mat.setdiag(0) #Removing the diagonal count from the calculation

        #Creating the base matrix for calculating mutual information
        mutual_i_mat = co_occurrence_mat.dot(p_mat_inv)
        mutual_i_mat = p_mat_inv.dot(mutual_i_mat)

        mutual_i_mat.data = np.log2(mutual_i_mat.data)/np.log2(log_base)

        #With the distributions defined some information can be negative.
        #These are set to zero.
        mutual_i_mat.data[mutual_i_mat.data <0] = 0

        #With this we have a series of weights for each term and pair of terms
        #Modifying p_mat to refer to information (without making a new copy)
        p_mat.data = -np.log2(p_mat.data)/np.log2(log_base)

        # #Adding in 'rarity' weight to doc-term mat
        doc_term_mat = doc_term_mat.dot(p_mat)

        # #Multiplying this by the mutual information to get the context sum per document
        context_mat = count_mat.dot(mutual_i_mat)

        # #Using element-wise multiplication to merge the two matrices
        doc_term_mat = doc_term_mat.multiply(context_mat)

        #Normalizing so that the L2 norm is one
        doc_term_mat = normalize(doc_term_mat, 'l2', axis = 1 )

        vocab = cv.get_feature_names_out()
                
        pickle.dump([doc_term_mat, vocab], open( vectorized_topic_location, "wb" ) ) #Saves the file.
        
    
    else: 
        tfidf_vect = TfidfVectorizer(min_df = min_word_count) # Creates the vectorizer instance
        
        word_tfidf = tfidf_vect.fit_transform(corpus) #vectorizes the corpus
        tfidf_word_list = tfidf_vect.get_feature_names() #Gets the feature names
        pickle.dump([word_tfidf, tfidf_word_list], open( vectorized_topic_location, "wb" ) ) #Saves the file.
    
    
    
    
    
    

