#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:31:48 2019

@author: Bill Konyk

A class that collects the NMF processing techniques demonstrated in http://dx.doi.org/10.13140/RG.2.2.25394.73922
"""

import tanagra.NMF_Analysis_Functions as naf
import tanagra.NMF_Preprocess_Functions as nppf
import tanagra.NMF_Plot_Functions as npf
import tanagra.NMF_Topic_Analysis_Functions as ntaf

import pickle
import os  
import numpy as np  
import pandas as pd

class NMF_Analysis:
    """
    A class that provides all the methods needed to automate an NMF analysis workflow.
    """
        
    def __init__(self, base_location, run_name, k_topics_list, *args, engine = None, **kwargs):
        
        """
        Initial creation of the NMF_Analysis method.
        Will create necessary folders.

        Parameters
        ----------
        base_location : str
            The root where all files will be stored
        run_name : str
            A specific name for the run, added to the base_location to build a path for files created
        k_topics_list : list
            Integer values for the number of topics that will (or have) been analyzed. Included so that the class has access to the names of files
        
        
        Returns
        -------
        none
        
        """
        
        #Names of folders
        self.data_folder = '{}/{}/Data/'.format(base_location, run_name)
        self.plot_folder = '{}/{}/Plots/'.format(base_location, run_name) #The location to save all plots
        self.run_name = run_name #The name of the run, used to name tables and files.
        
        self.doc_term_filename = self.data_folder + self.run_name + '_doc_word_tfidf.pkl'#file location of the vectorized text
        self.corpus_df_filename =  self.data_folder + self.run_name + '_corpus_df.pkl'#file location of a dataframe containing the corpus rather than the raw text.
        self.w2v_filename = self.data_folder+ self.run_name + '_w2v_skipgram.model' #The location of the word 2 vec skipgram model for the corpus
        
        self.model_data_folder = self.data_folder + 'Topic_Models/'
        self.topic_data_folder = self.data_folder + 'Topic_Assignments/'
        
        self.k_topics = k_topics_list
        self.h_filenames = [self.h_filename(num_topics) for num_topics in self.k_topics]
        self.assigned_filenames = [self.doc_topic_filename(num_topics) for num_topics in self.k_topics]
        
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            print('Created folder to save NMF models')
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
            print('Created folder to save plots')
        if not os.path.exists(self.model_data_folder):
            os.makedirs(self.model_data_folder)
            print('Created folder to save models')
        if not os.path.exists(self.topic_data_folder):
            os.makedirs(self.topic_data_folder)
            print('Created folder to save topics')
    
    #This method performs the preprocessing steps            
    def vectorize_data(self, text_df, *args, custom_stopwords = [], text_key = 'text', find_bigrams = True,
                        gensim_min_count= 10, gensim_threshold = 20, 
                        remove_text = False,
                        tfidf_min_word_count = 4, **kwargs):
        
        """
        Converts text data from a pandas dataframe containing the raw texts to analyze into a vectorized, machine friendly form.
        
        This includes several preprocessing steps:
                -Lemmatizing the text (i.e. converting his/him/he to he)
                -Removing stopwords (i.e. removing 'the')
                -Identifying bigrams (i.e. "United States")
                -Creating a word-2-vec model for future coherence analysis
                -Vectorizing the text using the tf-idf schemes
        
        Parameters
        ----------
        text_df : pandas Dataframe
            A dataframe containing all the text to analyze on different columns.
        cuom_stopwords : list
            Stopwords to remove from processing
        text_key : str
            Column name containing the text. Default is "text"
        find_bigrams: bool
            Allows one to turn off the bigram identification if desired
        gensim_min_count : int
            Minimum number of times two words must appear together for gensim to declare that it is a bigram
        gensim_threshold : int
            Threshold for gensim's bigram identification routine
        remove_text : bool
            If set true will remove the raw text from the output dataframe. Prevents duplication of data if the dataset is large
        tfidf_min_word_count : int
            Will remove words appearing fewer times than this value
        
        
        Returns
        -------
        text_df : pandas Dataframe
            Returns the dataframe with two new colums:
                -doc_id: an integer to uniquely identify the document for future processing
                -corpus: the cleaned version of the text after all preprocessing has been completed
        
        """
        
        try:
            text_df = pickle.load(open( self.corpus_df_filename, "rb" ))
            print('Corpus Loaded... Preprocessing Complete')
            
        except FileNotFoundError:
            print('Processing text')
            stopword_list = nppf.define_stopwords(custom_stopwords) #Defines a list of english stopwords using the NLTK package.
            
            corpus = nppf.create_corpus(text_df, stopword_list, text_key) # Creates the corpus by removing stopwords and lemmatizing with nltk
            print('    Text cleaned, corpus created')            
            
            if find_bigrams == True: #if find_bigrams is true this will attempt to identify bigrams in the text and will replace the original word with the bigram
                corpus, gensim_sentences = nppf.identify_bigrams(corpus, gensim_min_count, gensim_threshold)
                print('    Bigrams identified')
            else:
                gensim_sentences = nppf.create_gensim_sentences(corpus) #converts the dataset to a tokenized form for use with gensim
                
            nppf.create_w2v_skipgram_model(gensim_sentences, self.w2v_filename, 1) #this will create and save a skipgram word-2-vec model
            print('    Word2Vec model created')
            
            nppf.vectorize_corpus(corpus, self.doc_term_filename, tfidf_min_word_count) #this will save a vectorized model of the corpus
            print('    Corpus vectorized')
            
            text_df['corpus'] = corpus 
            
            text_df['doc_id'] = [i for i in range(len(text_df))]
            
            if remove_text:
                text_df.drop(text_key, axis = 1, inplace = True) #Removing the raw text to save on storage space
            
            pickle.dump(text_df, open( self.corpus_df_filename, "wb" ) ) #This will dump out a raw copy of the dataset to disk
        
            print('Corpus Created and Saved... Preprocessing Complete')
            
        return text_df
    
    #Returns the H filename
    def h_filename(self, num_topics):
        """
        Defines the name of the H matrix file
        
        Parameters
        ----------
        num_topics : int
            Number of topics being identified by NMF
            
        Returns
        -------
        filename : str
            Name of the H matrix file
        """

        return self.model_data_folder + self.run_name + '_'+str(num_topics).zfill(4) + "_topics_ensemble_model.pkl"
    
    def w_filename(self, num_topics):
        """
        Defines the name of the W matrix file
        
        Parameters
        ----------
        num_topics : int
            Number of topics being identified by NMF
            
        Returns
        -------
        filename : str
            Name of the W matrix file
        """
        
        return self.model_data_folder + self.run_name + '_'+str(num_topics).zfill(4) + "_W_ensemble_model.pkl"
      
        
    def run_analysis(self, num_folds, num_runs):
        
        """
        Method to run the NMF analysis process
        
        Parameters
        ----------
        num_folds : int
            Number of partitions into which the corpus will be divided
        num_runs : int
            Number of times to repeat the process
            
        Returns
        -------
        None
        """
        
        [doc_term_matrix, tfidf_word_list] = pickle.load(open(self.doc_term_filename, "rb" ) )
                
        num_topics = len(self.k_topics)

        for num_topics in self.k_topics:
            
            try:
                open(self.h_filename(num_topics), 'rb')
                print('NMF Matrices for ', num_topics,' already run.')
                
            except FileNotFoundError:
                
                ensemble_W, ensemble_H = naf.run_ensemble_NMF_strategy(num_topics, num_folds, num_runs, doc_term_matrix)
                
                pickle.dump((num_topics, ensemble_H), open(self.h_filename(num_topics), "wb" ) )
                pickle.dump(ensemble_W, open(self.w_filename(num_topics), "wb" ) )

                print('NMF Matrices for ', num_topics,' completed.')
                
                
    def doc_topic_filename(self, num_topics):
        """
        Defines the name of the doc-topic matrix file
        
        Parameters
        ----------
        num_topics : int
            Number of topics being identified by NMF
            
        Returns
        -------
        filename : str
            Name of the doc_topic matrix file
        """
        
        return self.topic_data_folder+'{}_doc_topic_mat_{}_topics.pkl'.format(self.run_name, num_topics)
                
    #Assigns the topics for anything in the k_topics list
    #Can pass a number of kwargs into the function in order to modify the behavior of the assignment process
    def assign_topics(self, angle_threshold, **kwargs):
        """
        Assigns topics...
        
        Parameters
        ----------
        num_topics : int
            Number of topics being identified by NMF
            
        Returns
        -------
        filename : str
            Name of the W matrix file
        """
        
        self.angle_threshold = angle_threshold
        
        print('Assigning Topics')
        
        for num_topics in self.k_topics:
            print('    Assigning topic {}'.format(num_topics))
            h_file = self.h_filename(num_topics)
            
            final_weights, H, entropy, invalid_docs = ntaf.assign_topics(angle_threshold, h_file, self.doc_term_filename, **kwargs)
            
            pickle.dump((final_weights, H, entropy, invalid_docs), open(self.doc_topic_filename(num_topics), 'wb') )
            
    
    '''
    Plot Functions
    '''
            
            
    def plot_topic_assignment_scan(self, *args, 
                                   min_angle = 0, 
                                   max_angle = 80,
                                   num_samples = 50,
                                   **kwargs):
        
        angle_list = np.linspace(min_angle, max_angle, num_samples)
        
        npf.plot_topic_assignment_scan(self.plot_folder, self.data_folder, self.run_name, self.h_filenames, self.doc_term_filename, angle_list)

            
            
    def plot_assigned_statistics(self, **kwargs):
        
        self.plot_verbosity()
        self.plot_topic_angles()
        self.plot_topic_coherence()
    
    def plot_topic_coherence(self, *args, **kwargs):
        
        npf.plot_self_coherence(self.run_name, self.data_folder, self.plot_folder, self.assigned_filenames, self.w2v_filename, self.doc_term_filename, **kwargs)
        
    def plot_verbosity(self, *args, **kwargs):  
        
        npf.plot_verbosity(self.run_name, self.data_folder, self.plot_folder, self.assigned_filenames, **kwargs)
        
    def plot_topic_angles(self, *args, **kwargs):  
        
        npf.plot_topic_angles(self.run_name, self.data_folder, self.plot_folder, self.assigned_filenames, **kwargs)
        
    def plot_topic_scan(self, **kwargs):
        
        npf.plot_topic_scan(self.run_name, self.assigned_filenames, self.data_folder, self.plot_folder, self.doc_term_filename, **kwargs)
        print('Topic Scan Completed')

    def plot_topic_reports(self, k_topics_to_plot, num_terms, **kwargs):
        
        h_filenames = [self.doc_topic_filename(num_topics) for num_topics in k_topics_to_plot]
        
        npf.plot_topic_report(self.run_name, h_filenames, self.doc_term_filename, self.plot_folder, self.data_folder, num_terms)
        print('Topic Reports Created')
        
    '''
    General Utilities
    '''    
        
    def load_corpus_df(self):
        return pickle.load(open(self.corpus_df_filename, 'rb'))
    
    
    def load_doc_term_and_vocab(self):
        
        doc_term_mat, vocab = pickle.load(open(self.doc_term_filename, 'rb'))
        
        return doc_term_mat, vocab
                                          
    def load_topic_name_csv(self, topic_name_filename):
        
        topic_name_df = pd.read_csv(topic_name_filename, engine='python')
        
        self.topic_name_df = topic_name_df
    
        
    '''
    Functions to output data in user and plot friendly ways
    '''
        
    def print_tabular_format(self, k_topics, *args, 
                       print_tab_form = True, 
                       overwrite_saved_data = False, 
                       **kwargs):
                        
        data_filenames = [self.doc_topic_filename(num_topics) for num_topics in k_topics]
        
        return ntaf.tabularize_data(self.data_folder, self.run_name, self.corpus_df_filename, data_filenames, self.topic_name_df,
                                    print_tab_form = print_tab_form,
                                    overwrite_saved_data = overwrite_saved_data
                                    )
    
    def print_topic_assignments(self, num_topics, 
                               top_words = 5, top_topics=2, columns_to_include = ['text']):
        
        """
        Prints out the topic assignemnts for each doc_topic_mat.
        
        Parameters
        ----------
        num_topics : int
            Prints out the analysis using this number of topics
        top_words : int
            Number of top words to print
        top_topics : int
            Number of top topics to print
        columns_to_include : list
            Columns to include. Default is to include 'text'.
            
        Returns
        -------
        assigned_df : pandas dataframe
            Dataframe containing the top words, top topics of the text along with any other columns specified
        """
        
        doc_term_mat, vocab = self.load_doc_term_and_vocab()
        
        filename = self.data_folder + '{}_topic_assignment_data.csv'.format(self.run_name)
        
        (doc_topic_mat, H, entropy, invalid_docs) = pickle.load(open(self.doc_topic_filename(num_topics), 'rb'))
        
        assigned_df = ntaf.generate_topic_assignments(self.topic_name_df, doc_term_mat, vocab, doc_topic_mat, top_words, top_topics)
        
        print(assigned_df.columns)
        
        corpus_df = pickle.load(open(self.corpus_df_filename, 'rb'))
        
        print(corpus_df.columns)
        
        assigned_df = assigned_df.merge(corpus_df[['doc_id'] + columns_to_include], on = ['doc_id'])
        
        #Identifying invalid docs:
        assigned_df['valid_assignment'] = 1
        assigned_df.loc[assigned_df['doc_id'].isin(invalid_docs), 'valid_assignment'] = 0
        
        assigned_df.to_csv(filename, index = False)
            
        return assigned_df
    
        
        
                
