# -*- coding: utf-8 -*-
"""
Created on Fri May 14 01:53:28 2021

@author: konyk
"""

import pickle
import numpy as np
import re
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import normalize


def calc_angle( v1, v2):
    """
    Calculates the angle between two vectors
    
    Parameters
    ----------
    v1 : np array
        Vector 1
    v2 : np array
        Vector 2
        
    Returns
    -------
    angle : float
        Angle between the vectors
    """
    
    norm = np.linalg.norm(v1)*np.linalg.norm(v2)
    
    if norm == 0:
        return 0 
    else:
        dot = np.dot(v1, v2)/(norm)
        return np.arccos(dot)



def cull_with_angle(H, threshold, *args, units = 'radians', min_words = 15, **kwargs):
    """
    Culls a matrix using an angle threshold, returns the indices removed, and renormalizes the matrix.
    
    The idea is that the H matrix is made up of N orthogonal vectors in the term space.
    The lowest terms are removed until the angle between the original topic (row of H) and a new topic is at a specified threshold.
    
    This removes some of the tail present in the topic matrix. Choosing a larger angle will remove more of the rows. 

    
    Parameters
    ----------
    H : matrix
        H Matrix to process
    threshold: float
        Angle below which values will be removed
    units : string
        Units that the threshold is in. Default is radians, accepts 'degrees' as well
        
    Returns
    -------
    doc_df : pandas dataframe
        Dataframe containing the word and character counts
    """
    
    #Creates a 
    raw_list = []
    
    if units == 'degrees':
        threshold *= np.pi/180
    
    #Defines the new h matrix
    new_H = np.zeros(H.shape)
    
    #For each topic (row in H)
    for topic in range(H.shape[0]):
        h_indices = H[topic].argsort()[::-1] #Sorts the indices by value
        
        h_vals = [H[topic, i] for i in h_indices] #Rearranges H in terms of values
        
        raw_list.append(h_vals) #Stores the original values
        
        test_mat = np.zeros(len(h_vals)) #Creates a test matrix

        #Adds values one by one to the test matrix, starting with the largest
        for j in range(len(h_vals)):
            test_mat[h_indices[j]] = h_vals[j]
            
            term = h_indices[j]
            new_H[topic,term] =  H[topic,term]
            
            #If the angle between the new topic and the original one is less than the threshold the algorithm halts, having found the appropriate number of values
            #Program will also continue until there are at least min_words number of terms included to prevent the process from cutting things down to a single word or phrase.
            if calc_angle(H[topic], test_mat)<threshold and j+1  >= min_words: 
                break
    
    #Renormalizes H        
    new_H = normalize(new_H, norm = 'l2', axis = 1)
        
    return new_H , raw_list

def assign_topics(angle_threshold, h_file, dt_file, *args,
                  topics_to_remove = [], #Allows user to prevent the process from assigning any documents to particular topics
                  remove_constant = True, #Remove a constant from the comparison
                  max_topics = 3, #Default is to allow up to 3 topics
                  relative_threshold = .1, #Will remove any topics that are this ratio compared to the largest
                  angle_units = 'degrees', #Defines the units of the threshold being input. Default is degrees
                  num_entropy_bins = 1000, #Number of bins to use in calculating entropy
                  entropy_smoothing_sigma = 6, #Level of smoothing to apply on the information function
                  **kwargs):
    
    """
    Assigns topics to documents. There are a few steps in the process:
        -The number of terms in H can be reduced to 'focus' the topic. This is accomplished by removing terms until the overlap between the new topic and old topic is at the specificed angle threshold.
        -A constant can be removed based on entropy. An effective probability distribution is created from the nonzero doc-topic angles and a constant value in angle, corresponding to the information equaling the entropy, is removed.
        -The maximum number of topics to assign can be specified.
        -A relative threshold can be applied so that only larger topics are kept. For example, if two topics are assigned with values of .9 and .001, the second would be elmininated on a threshold of .1

    
    Parameters
    ----------
    angle_threshold : float
        Angle by which to reduce the topics in the H matrix
    h_file: String
        Filename for the H matrix
    dt_file : string
        Filename for the vectorized doc-term matrix
    topics_to_remove : list
        A list of integers corresponding to the topics to remove (rows in the H matrix). These assignments will be zeroed out.
    remove_constant : bool
        Will remove a constant angle based on an information/entropy criterion. Used to identify documents that do not fit the angles well
    max_topics : int
        Will assign at most this number of topics (possibly less)
    relative_threshold : float
        Value between 0 and 1, will remove topics assigned with a value less than this
    angle_units : string
        Accepts degrees or radians, used to specify the angle passed in by the angle_threshold
    num_entropy_bins : int
        Number of bins for calculating the effective probability distribution
    entropy_smoothing_sigma : int
        Level of smoothing to pass into a scipy gaussian_filter1d function
    
    Returns
    -------
    final_weights : matrix
        The final document-topic weighting. Values are set so that the total per document sums to 1. 
    H : matrix
        The final H matrix after removing terms. If any topics are removed, this will be reflected in H.
    entropy : float
        The entropy cutoff used    
    invalid_docs : list
        A list of indices in final_weights corresponding to invalid documents.
    """
    
    #Load the needed data
    (num_topics, H) = pickle.load(open(h_file,'rb'))
    doc_term_mat, vocab = pickle.load(open(dt_file,'rb'))
    
    doc_term_mat = normalize(doc_term_mat, norm = 'l2', axis = 1)    #Normalizes the doc-term matrix
    try:
        doc_term_mat = doc_term_mat.toarray() #Converting to a format that can be used (needed because this can be a sparse matrix)
    except:
        pass

    #Modify H by removing terms until the angle with the original is at the threshold
    H , raw_list = cull_with_angle(H, angle_threshold, units = angle_units)
    
    #Because the doc-term matrix and H are both normalized, the angle between topics and documents are easy to calculate using matrix multiplication.
    doc_topic_mat = np.matmul(doc_term_mat, np.transpose(H))
    
    #From this creating a matrix consisting of the angles between each of the matrices. 
    #As H and dt are normalized, arccos is sufficient to capture the angle
    #Note that values are scaled so that 0 corresponds to no difference whereas pi/2 corresponds to nearly the same.
    #The point of this scaling is that large values should correspond to good data (even though the angle between two parallel vectors is actually zero)
    doc_topic_angles = np.pi/2 - np.arccos(doc_topic_mat)
    
    #Will remove a constant value based on entropy. Empirically, however, this tends to be low so it doesn't seem to contribute much. 
    #It is likely needed for larger numbers of topics.
    if remove_constant:
        
        #Creating an effective, discrete probability distribution
        data = np.ravel(doc_topic_angles)
        
        #Removing zero values
        data = data[data>0]
        
        #Using entropy to identify a cutoff value           
        bins = np.linspace(0, np.pi/2, num_entropy_bins)
        
        #Creating the histogram of data
        total_hist, xvals = np.histogram(data, bins = bins)
        
        #Normalizing the calculated histogram to create an effective probability distribution
        norm_hist = total_hist/total_hist.sum()
        
        #Removing empty bins so that the log function in entropy won't throw errors
        nonzero_indices = np.where(norm_hist>0)[0]
        norm_hist = norm_hist[nonzero_indices]
        x_vals = bins[nonzero_indices]
        
        #Calculating entropy (i.e. the expectation value of probability)
        entropy = sum(-norm_hist*np.log2(norm_hist))
        
        #Calculating the information content of each bin and smoothing
        information = -np.log2(norm_hist)
        information = gaussian_filter1d(information, entropy_smoothing_sigma)
        
        #The bin of the cutoff where information becomes greater than the entropy
        cutoff_index = np.where(information <= entropy)[0].max() +1
        
        #The angle value of the cutoff
        angle_cutoff = x_vals[cutoff_index]
        
        #Identifying the location of the maximum topic for assigning invalid documents (ones with all angles below the threshold) 
        max_topic_indices = [np.argmax(doc_topic_angles[doc]) for doc in range(doc_topic_angles.shape[0])]    
        
        #Subtracing the cutoff value from each weight
        doc_topic_angles -= angle_cutoff
        
        #Removing any negative values
        doc_topic_angles[doc_topic_angles<0] = 0
            
        #Identifying indices where all angles are negative (no information on topic assignment)
        invalid_docs = np.where(np.max(doc_topic_angles, axis =1)<=0)[0]  
        
        #Assigns the invalid doc to the topic that best matches (highest angle)
        for doc in invalid_docs:
            max_topic = max_topic_indices[doc]
            doc_topic_angles[doc, max_topic] = 1
            
    else:
        invalid_docs = []
        entropy = 0
       
    #Defines the rank of each angle 
    doc_topic_rank = np.argsort(doc_topic_angles, axis = 1)[:,::-1]
    
    #Removing user-specificed topics from processing
    for topic in topics_to_remove:
        H[topic] = 0
        doc_topic_angles[:, topic] = 0
    
    #Removing anything but the top n topics
    for i, angles in enumerate(doc_topic_rank[:, max_topics :]):
        doc_topic_angles[i, angles] = 0
    
    #Normalizing the matrix, creating the final weighting
    final_weights = normalize(doc_topic_angles, norm = 'l1', axis = 1)

    #Removing any assignments less than a relative threshold
    if relative_threshold > 0:
        final_weights[final_weights < relative_threshold] = 0
        
        final_weights = normalize(final_weights, norm = 'l1', axis = 1) 
    
    return final_weights, H, entropy, invalid_docs


def count_words(doc_df, text_column):
    """
    Adds the columns "word_count" and "character_count" to the dataframe
    
    Parameters
    ----------
    doc_df : pandas dataframe
        Dataframe containing the text to process
    text_column: str
        Column to count words and characters
        
    Returns
    -------
    doc_df : pandas dataframe
        Dataframe containing the word and character counts
    """
    
    word_count = []
    character_count = []
    
    for text in doc_df[text_column].values:
        text = re.sub(r'[^\w\s]','',text) #Removes anything that is not a space or a character
        character_count.append(len(re.sub(r'\s','',text))) #Removing spaces to get all characters
        
        #Counting words by spaces
        tokens = text.split()
        word_count.append(len([word for word in tokens if len(word)>0 ])) #Removing any 'words' that are empty
        
    doc_df['word_count'] = word_count
    doc_df['character_count'] = character_count
    
    return doc_df
    

def tabularize_data(data_folder, run_name, corpus_df_filename, data_filenames, topic_name_df, *args, 
                    print_tab_form = False, 
                    overwrite_saved_data = False, 
                    **kwargs):
    
    tab_filename = data_folder + '{}_Tabular_df.pkl'.format(run_name)
    
    try:
        if overwrite_saved_data == True:
            raise IndexError
        else:
            tabular_df = pickle.load(open(tab_filename, 'rb'))
            
            print('Tabular DF loaded')
    
    except:
        doc_df  = pickle.load(open(corpus_df_filename, 'rb'))
        
        doc_df = count_words(doc_df, 'corpus')
        
        doc_id_list = []
        topic_title_list = []
        k_topics_list = []
        topic_nbr_list = []
        topic_weight_list = []
        word_count_list = []
        character_count_list = [] 
                    
        for filename in data_filenames: 
            
                (doc_topic_mat, H, entropy, invalid_docs)  = pickle.load(open(filename,'rb'))
                num_topics = doc_topic_mat.shape[1]
                
                print('Tabluarizing {} topics'.format(num_topics))
                
                for doc in range(doc_topic_mat.shape[0]):
                    for topic_num in range(doc_topic_mat.shape[1]):
                        
                        
                        
                        if doc_topic_mat[doc, topic_num] > 0.01:
                            try:
                                word_count, char_count = doc_df[doc_df['doc_id'] == doc][['word_count', 'character_count']].values[0]
                                topic_title = (topic_name_df[(topic_name_df['topic_nbr']== topic_num+1) & (topic_name_df['k_topics']== num_topics) ]['topic_title'].values[0])

                                doc_id_list.append(doc)
                                k_topics_list.append(num_topics)
                                topic_nbr_list.append(topic_num +1)
                                topic_weight_list.append(doc_topic_mat[doc, topic_num])
                                word_count_list.append(word_count)
                                character_count_list.append(char_count)
                                topic_title_list.append(topic_title)
                                
                            except IndexError:
                                pass
                        
        tabular_df = pd.DataFrame({'doc_id': doc_id_list,
                                    'word_count' : word_count_list,
                                    'char_count' : character_count_list,
                                    'topic_weight' : topic_weight_list,
                                    'k_topics' : k_topics_list,
                                    'topic_nbr' : topic_nbr_list,
                                    'topic_title': topic_title_list})
    
        tabular_df.sort_values(['k_topics','doc_id','topic_weight'], inplace = True)
        
        #Identifying invalid docs:
        tabular_df['valid_assignment'] = 1
        tabular_df.loc[tabular_df['doc_id'].isin(invalid_docs), 'valid_assignment'] = 0
        
        tabular_df.to_csv(tab_filename, index = False)
                    
    return tabular_df


#Takes in a matrix and outputs a ranked matrix where the rows are the values of the original matrix's rows reordered, and a second matrix with the indices of the original matrix for these reordered values.
def rank_matrix(mat):
    
    """
    Rearranges a matrix so that the rightmost value in each column is the largest one. 
    Outputs a mapping of the indices to be able to recover the original form.
    
    Parameters
    ----------
    mat : matrix
        Matrix to rank
        
    Returns
    -------
    rank_mat : matrix
        Matrix with rows rearranged so that highest values is rightmost
    index_mat : matrix
        Index of original values
    """
    
    #Generate blank matrices
    rank_mat = np.zeros(mat.shape)
    index_map = np.zeros(mat.shape)
    
    #Iterate over each row
    for row in range(mat.shape[0]): 
        index_list = np.argsort(mat[row])[::-1] #Sort the indices by rank
        row_vals = [mat[row, i] for i in index_list] #Use this sorted list to rearrange the values
        
        #Recording the values in terms of their rank within the document
        for val in range(len(row_vals)):
            rank_mat[row, val] = row_vals[val]
            index_map[row,val] = index_list[val]
    
    return rank_mat, index_map




def generate_topic_assignments(topic_name_df, doc_term_mat, vocab, doc_topic_mat, 
                               top_words, top_topics):
    """
    Generates a dataframe with all the top topics and words for each document for analysis and plotting purposes
    
    Parameters
    ----------
    topic_name_df : pandas dataframe
        Dataframe containing the topic names. 
    doc_term_mat : matrix
        Document-term matrix
    vocab : list
        List of the vocabulary
    doc_topic_mat : matrix
        Document-Topic matrix
    top_words : int
        Number of top words to print
    top_topics : int
        Number of top topics to print
        
    Returns
    -------
    final_df : pandas dataframe
        Dataframe containing the top words, top topics of the text
    """
    
    
    #If the doc_term_mat is a sparse matrix, this will convert it into an array for processing
    try:
        doc_term_mat = doc_term_mat.toarray()
    except:
        pass
    
    #Creating a term ranking for the corpus in order to determine the top words to display
    doc_term_rank_mat, doc_term_index_map = rank_matrix(doc_term_mat)
        
    top_word_list = []
    #Appending the top words for each document to the top_word_list        
    for doc in range(doc_term_mat.shape[0]):
        word_list = []
        for word in range(top_words):
            
            vocab[int(doc_term_index_map[doc,word])]
            
            word_list.append(vocab[int(doc_term_index_map[doc,word])])
        top_word_list.append(', '.join(word_list))
        
    
    num_topics = doc_topic_mat.shape[1]
    
    data = {}
    data['doc_id'] = [doc for doc in range(doc_term_mat.shape[0])]
    data['top_words'] = top_word_list
    
    for i in range(top_topics):
        data['topic_{}'.format(i+1)] = []
    for i in range(top_topics):
        data['w_val_{}'.format(i+1)] = []
    
    for doc in range(doc_topic_mat.shape[0]):
        
        index_list = doc_topic_mat[doc].argsort()[::-1]
        row_vals = [doc_topic_mat[doc, i] for i in index_list]
        
        for i in range(top_topics):
            if row_vals[i]>.01:
                data['topic_{}'.format(i+1)].append(topic_name_df[(topic_name_df['topic_nbr']== index_list[i]+1) & (topic_name_df['k_topics']== num_topics) ]['topic_title'].values[0])
                data['w_val_{}'.format(i+1)].append(round(row_vals[i],2))
            else:
                data['topic_{}'.format(i+1)].append('')
                data['w_val_{}'.format(i+1)].append(0.0)
    
    final_df = pd.DataFrame(data)
    
    final_df = final_df.sort_values(by = ['topic_1','w_val_1'], ascending = False)
                
    print('Finished {} topics.'.format(num_topics))
    
    return final_df


# def generate_text_assignments(doc_df, doc_term_mat, vocab, doc_topic_mat, 
#                             top_words, top_topics):
    
#     topic_assignment_df = generate_topic_assignments(doc_df, doc_term_mat, vocab, doc_topic_mat, top_words, top_topics)
    
#     assigned_df = ta.print_topic_assignments()

#     #Creating a document with the actual text included, since the titles were incorrectly assigned
#     assigned_df = assigned_df.merge(doc_df[['doc_id','text']], on = ['doc_id'])
    
    
#     assigned_df = assigned_df[['doc_id',  'topic_1', 'topic_2', 'topic_3', 'w_val_1', 'w_val_2', 'w_val_3', 'top_words', 'text']]
#     assigned_df = assigned_df.merge(doc_df[['doc_id','year']], on = 'doc_id', how = 'inner')
    
#     assigned_df.to_csv(ta.print_save_path +'Text Assignments for {} Topics.csv'.format(combined_H.shape[0]), index = False)
    
    
# def generate_topic_summary():
    
#         def print_topic_summary(self):
#         try:
#             len(self.tabular_df)
                
#         except AttributeError:
#             self.tabularize_data()
            
#         self.tabular_df['res_count'] = 1
#         df = pd.DataFrame(self.tabular_df.groupby(['k_topics', 'topic_title']).agg(
#                                          {'word_count':['sum'],
#                                           'char_count':['sum'],
#                                           'res_count':['sum'],
#                                           'topic_weight':['sum']})).reset_index()
    
#         summary_df = pd.DataFrame({'k_topics': df['k_topics'],
#                                    'topic_title' : df['topic_title'],
#                                    'word_count' : df['word_count']['sum'] * df['topic_weight']['sum'],
#                                    'char_count' : df['char_count']['sum'] * df['topic_weight']['sum'],
#                                    'topic_weight' : df['topic_weight']['sum'],
#                                    'num_resolutions' : df['res_count']['sum']})    
    
#         summary_df.sort_values(['k_topics','topic_title'], inplace = True)
    
#         k_topic_list = summary_df['k_topics'].unique()
                
#         self.makedir(self.print_save_path+'/Topic Title Summaries/', message = 'Created folder to save topic title summaries')        
        
#         for num_topics in k_topic_list:
#             print_df = summary_df[summary_df['k_topics'] == num_topics].copy()
            
#             print_df['word_percent'] = round(print_df['word_count']/print_df['word_count'].sum()*100,2)
#             print_df['char_percent'] = round(print_df['char_count']/print_df['char_count'].sum()*100,2)
#             print_df['topic_percent'] = round(print_df['topic_weight']/print_df['topic_weight'].sum()*100,2)
#             print_df['num_resolutions_percent'] = round(print_df['num_resolutions']/print_df['num_resolutions'].sum()*100,2)
            
#             print_df.sort_values(['topic_percent'], inplace = True)
            
#             print_df.to_csv(self.print_save_path+'/Topic Title Summaries/Topic Title Summary {} Topics.csv'.format(num_topics), index = False)
            
