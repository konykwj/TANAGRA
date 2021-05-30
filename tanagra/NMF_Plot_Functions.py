#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 02:37:37 2019

@author: konykwj

Contains all the functions necessary to execute plot commands in the NMF_Analysis class
"""
import pickle
from os import listdir
from os.path import isfile, join
import os
import gensim
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import ticker

from sklearn.cluster import KMeans

from NMF_Analysis.NMF_Analysis_Functions import b_mat
from NMF_Analysis.NMF_Topic_Analysis_Functions import assign_topics


matplotlib.use('Agg')

def get_descriptor( all_terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        all_terms[term_index]
        top_terms.append( all_terms[term_index] )
    return top_terms

def get_top_values( all_terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_values = []    
    for term_value in top_indices[0:top]:
        top_values.append(H[topic_index,term_value])
    return top_values

def calculate_self_coherence(num_topics, verbosity_list, w2v_model, H, vocab):
    #This provides a list of the coherence for each topic
    
    term_rankings = []
    for q_topic in range(num_topics):
        term_rankings.append(get_descriptor( vocab, H, q_topic, verbosity_list[q_topic])) 
    
    self_coh = np.zeros([num_topics])
    
    for topic_i in range(len(term_rankings)):
        terms = term_rankings[topic_i]
        pairs = []
        for p in range(len(terms)):
            for q in range(p+1):
                pairs.append((terms[p],terms[q]))
                
        pair_scores =[]
        for topic_ind in range(len(pairs)):
            pair_scores.append(w2v_model.wv.similarity(pairs[topic_ind][0], pairs[topic_ind][1]) )
        self_coh[topic_i] = np.mean(pair_scores)
    
    return self_coh


def create_topic(word_list, n):
    return ', '.join(word_list[0:n])


def calc_angle(v1, v2):
    return np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))


# def verbosity(H, threshold, *args, units = 'degrees', **kwargs):
    
#     if units == 'degrees':
#         threshold *= np.pi/180
    
#     v_list = []
#     for topic in range(H.shape[0]):
#         h_indices = H[topic].argsort()[::-1]
        
#         h_vals = [H[topic, i] for i in h_indices]
#         test_mat = np.zeros(len(h_vals))
#         print('::::::::::::::::::::::::::;')
#         for j in range(len(h_vals)):
#             test_mat[h_indices[j]] = h_vals[j]
            
#             angle = calc_angle(H[topic],test_mat)
#             print(angle , j, threshold, h_vals[j], max(h_vals))
#             if angle < threshold:
#                 break
#         v_list.append(max(1,j-1))
        
#     return v_list
    
def verbosity(H, threshold):
    
    v_list = []
    for topic in range(H.shape[0]):
        h_indices = H[topic].argsort()[::-1]
        
        h_vals = [H[topic, i] for i in h_indices]
        h_vals /= max(h_vals)
        
        verb = np.where(h_vals <  threshold)[0].min()
        
        # test_mat = np.zeros(len(h_vals))

        # for j in range(len(h_vals)):
        #     test_mat[h_indices[j]] = h_vals[j]
            
            
        #     if summ > threshold:
        #         break
        v_list.append(verb)
        
    return v_list

def doc_norm_fun(W, doc_index):
    
    try:
        return 1/sum(W[doc_index])
    except FloatingPointError:
        print('********')
        print(np.shape(W), doc_index)
        print(sum(W[doc_index]))
        
        return 1

def w_sum_fun(W, topic_index):
    return W[:,topic_index].sum()


def plot_scatter_line(data_df, plot_folder, filename, x_column, y_column, x_label, y_label, color_list, *args,
                      alpha_val = .3, fig_size = (9,7), **kwargs):
       
    fig, ax1 = plt.subplots(figsize= fig_size)
    
    ax1.scatter(data_df[x_column].values, data_df[y_column].values, color = color_list[0], alpha = alpha_val)
    
    #Calc stats
    stat_df = pd.DataFrame(data_df.groupby(x_column).agg({y_column:['max','min','mean']})).reset_index()
    
    # print(stat_df.columns)
    
    ax1.fill_between(stat_df[x_column].values, stat_df[y_column]['min'].values, stat_df[y_column]['max'].values,
        alpha=0.1, edgecolor=color_list[1], facecolor=color_list[2])
    
    #Calc average
    
    ax1.plot(stat_df[x_column].values, stat_df[y_column]['mean'].values, 'k')
    
    ax1.set_ylabel(y_label)
        
    # plt.title(title)
    ax1.set_xlabel(x_label)
    
    ax1.grid(which='major', axis='both', linestyle='--')
    
    fig.tight_layout()
    fig.savefig(plot_folder + filename, dpi = 150)

    plt.close('all') 
    
    

#Plots a number of different topic statistics
def plot_self_coherence(run_name, data_folder, plot_folder, h_filenames, w2v_file, doc_term_filename, *args, 
                               num_terms = 150, filetype = '.png', **kwargs):
            
    print('Plotting Self-Coherence')
    
    #Pulls the word 2 vec model and tf-idf matrix created previously
    w2v_model = gensim.models.Word2Vec.load(w2v_file)
    
    [doc_term_mat, vocab] = pickle.load(open(doc_term_filename, "rb" ) )
    
    
    try:
        coherence_df = pickle.load(open(data_folder+'{}_self_coherence.pkl'.format(run_name), 'rb'))

    except FileNotFoundError:
        
        #Creating lists to hold the topic quality metrics
        
        scatter_topic_list = []
        scatter_coh = []
        scatter_topic_nbr_list = []
            
        for p, h_file in enumerate(h_filenames):
            
            (final_weights, H, entropy, invalid_docs) = pickle.load(open(h_file,'rb'))
            num_topics = H.shape[0]
                        
            # verbosity_list = verbosity(H, verbosity_value)
            verbosity_list = [num_terms for term in range(num_topics)]
            
            self_coh = calculate_self_coherence(num_topics, verbosity_list, w2v_model, H, vocab)
            
            topic_nbr = 1
            for coh in self_coh:
                scatter_coh.append(coh)
                scatter_topic_list.append(num_topics)
                scatter_topic_nbr_list.append(topic_nbr)
                topic_nbr += 1
                
            print('Coherence Plots', p/len(h_filenames)*100.,' % complete')
        
        
        coherence_df = pd.DataFrame({'k_topics':scatter_topic_list,
                                     'topic_nbr':scatter_topic_nbr_list,
                                     'coherence':scatter_coh})
        
        pickle.dump(coherence_df,open(data_folder+'{}_self_coherence.pkl'.format(run_name), 'wb'))
        
    '''
    Plotting self coherence
    '''
    
    plot_scatter_line(coherence_df, 
                      plot_folder, 
                      '{}_self_coherence{}'.format(run_name, filetype), 
                      'k_topics', 
                      'coherence', 
                      'Topic Number',
                      'Topic Coherence', 
                      ['#CC4F1B', '#CC4F1B', '#FF9848'])
              
def plot_verbosity(run_name, data_folder, plot_folder, h_filenames, *args, 
                               verbosity_value = .1, filetype = '.png', **kwargs):
    
    print('Plotting Verbosity')
    
    try:
        
        #Creating dataframes for future use
        verbosity_df = pickle.load(open(data_folder+'{}_verbosity.pkl'.format(run_name), 'rb'))

    except FileNotFoundError:
        
        #Creating lists to hold the topic quality metrics
        
        verbosity_vals = []
        verbosity_num_topics = []
        verbosity_topic_nbr = []
            
        for p, h_file in enumerate(h_filenames):
            
            (final_weights, H, entropy, invalid_docs) = pickle.load(open(h_file,'rb'))
                        
            verbosity_list = verbosity(H, verbosity_value) #convert verbosity to angle
            
            for topic_nbr, verb in enumerate(verbosity_list):
                print('**************')
                print(verb, topic_nbr)
                verbosity_vals.append(verb)
                verbosity_num_topics.append(H.shape[0])
                verbosity_topic_nbr.append(topic_nbr)
                topic_nbr += 1
            
        
        #Creating dataframes for future use
        verbosity_df = pd.DataFrame({'k_topics':verbosity_num_topics,
                                     'topic_nbr':verbosity_topic_nbr,
                                     'verbosity':verbosity_vals})
        
        # pickle.dump(verbosity_df, open(data_folder+'{}_verbosity.pkl'.format(run_name), 'wb'))
    
    '''
    Plotting verbosity
    '''
    
    plot_scatter_line(verbosity_df, 
                      plot_folder, 
                      '{}_verbosity{}'.format(run_name, filetype), 
                      'k_topics', 
                      'verbosity', 
                      'Topic Number',
                      'Topic Verbosity', 
                      ['#089FFF', '#1B2ACC', '#089FFF'])
    
    
    #Another list of colors; ['#C69C04', '#F5BF03', '#C69C04']
    
def plot_topic_angles(run_name, data_folder, plot_folder, h_filenames, *args, 
                               verbosity_value = .75, filetype = '.png', normalize_H = False, **kwargs):
    
    print('Plotting the distribution of angles between topics')
    '''
    Plotting the angle distributions
    
    Note that it is assumed that H is already normalized.
    '''
    angle_list = []
    num_topic_list = []
    
    for p, h_file in enumerate(h_filenames): 
        (final_weights, H, entropy, invalid_docs) = pickle.load(open(h_file,'rb'))
        
        #Used if H is not normalized
        if normalize_H:
            B,B_inv = b_mat(H)
            H = np.matmul(B,H)
                
        for topic in range(H.shape[0]):
            for topic2 in range(topic+1):
                if topic != topic2:
                    angle = (np.pi/2.0 - np.arccos(np.dot(H[topic], H[topic2])))/(np.pi/2.0)*100
                    num_topic_list.append(H.shape[0])
                    angle_list.append(angle)
                    
        
        
        
    angle_df = pd.DataFrame({'k_topics': num_topic_list,
                             'angle': angle_list})
        
    plot_scatter_line(angle_df, 
                      plot_folder, 
                      '{}_angle_distribution{}'.format(run_name, filetype), 
                      'k_topics', 
                      'angle', 
                      'Topic Number',
                      'Topic Angles', 
                      ['#3F7F4C', '#3F7F4C', '#7EFF99'])
            
    
    


def plot_topic_scan(run_name, h_filenames, data_save_path, plot_save_path, doc_term_filename):
    
    # path = nmf_model_path
    # filenames = [f for f in listdir(path) if isfile(join(path, f)) and 'all_H' not in f]
    # filenames.sort()
    
    # file_list = []
    # input_k_topic_list.sort()
    
    # for k_topics in input_k_topic_list:
    #     for file in filenames:
    #         if str(k_topics).zfill(4) in file:
    #             file_list.append(file)
    
    [doc_term_mat, vocab] = pickle.load(open( doc_term_filename, "rb" ) )    
    
    #lists for the dataframe that maps topics from one k to another
    angle_list = []
    topic_index_list = []
    next_topic_index_list = []
    k_topics_list = []
    next_k_topics_list = []
    
    #lists for the dataframe that will give the W values for each topic
    w_k_topic_list = []
    w_topic_index_list = []
    w_val_list = []
    w_topic_name = []
    
    for p in range(len(h_filenames)-1): 
        (W, H, entropy, invalid_docs) = pickle.load(open(h_filenames[p],'rb'))
        num_topics = H.shape[0]
        
        (W_p1, H_p1, entropy, invalid_docs) = pickle.load(open(h_filenames[p],'rb'))
        num_topics_p1 = H_p1.shape[0]
                        
        num_docs = W.shape[0]
    
        for topic in range(num_topics):
            w_k_topic_list.append(num_topics)
            w_topic_index_list.append(topic)
            w_val_list.append(w_sum_fun(W,topic))
            
            top_words = get_descriptor( vocab, H, topic, 3 )
            w_topic_name.append(' ,'.join(top_words))
            
            for topic_p1 in range(num_topics_p1):
                angle = np.arccos(np.dot(H[topic], H_p1[topic_p1]))
                k_topics_list.append(num_topics)
                angle_list.append(angle)        
                topic_index_list.append(topic)
                next_topic_index_list.append(topic_p1)
                next_k_topics_list.append(num_topics_p1)
            
    #Adding in the last topic
    (W, H, entropy, invalid_docs) = pickle.load(open(h_filenames[-1],'rb'))

    num_docs = W.shape[0]
    
    for topic in range(num_topics):
        w_k_topic_list.append(num_topics)
        w_topic_index_list.append(topic)
        w_val_list.append(w_sum_fun(W,topic))
        
        top_words = get_descriptor( vocab, H, topic, 3 )
        w_topic_name.append(' ,'.join(top_words))
        
    X = np.reshape(angle_list,(-1,1))
    X = np.nan_to_num(X)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)    
    
    topic_similarity_list = kmeans.labels_
    
    w_val_dataframe = pd.DataFrame({'k_topics':w_k_topic_list,
                                  'topic_index':w_topic_index_list,
                                  'w_value':w_val_list,
                                  'topic_name':w_topic_name})
        
    topic_similarity_dataframe = pd.DataFrame({'k_topics_i':k_topics_list,
                                  'k_topics_j':next_k_topics_list,
                                  'topic_index_i':topic_index_list,
                                  'topic_index_j':next_topic_index_list,
                                  'same_flag':topic_similarity_list,
                                  'angle':angle_list})
    
    w_val_dataframe = w_val_dataframe.fillna(0)
    topic_similarity_dataframe = topic_similarity_dataframe.fillna(0)
                
    w_val_dataframe.sort_values(['k_topics','topic_index'], inplace = True)
    
        
    #Adding in a test to ensure that kmeans is always correctly flagging the dataset with the smaller angle as 1
    dissimlar_mean = topic_similarity_dataframe[topic_similarity_dataframe['same_flag']==0]['angle'].mean()
    similar_mean = topic_similarity_dataframe[topic_similarity_dataframe['same_flag']==1]['angle'].mean()
    
    if dissimlar_mean < similar_mean:
        topic_similarity_dataframe['same_flag'] = [1-x for x in topic_similarity_list]
        
    pickle.dump(topic_similarity_dataframe, open( data_save_path+run_name+'_topic_similarity_data.pkl', "wb" ) )
    
    
    #Plotting the results of k_means in order to show the user that it indeed correctly identified the topic.
    #fig = plt.figure(figsize=(8,11))
    fig, ax1 = plt.subplots(figsize=(9, 7))
    
    ax1.scatter(k_topics_list, angle_list, alpha = .3, color = ['r' if label == 1 else 'b' for label in topic_similarity_dataframe['same_flag'].values])
        
    ax1.set_ylabel(r'$\theta$ between topics')
    
    plt.title('Topic Similarity Clustering Results')
    ax1.set_xlabel('Min k in Comparison')
    
    ax1.grid(which='major', axis='both', linestyle='--')

    fig.tight_layout()

    fig.savefig(plot_save_path + '{}_topic_angle_distributions.png'.format(run_name), dp1 = 300)
    
    plt.close('all') 
    
    #Creating a numpy matrix to hold all the information I need
    k_topic_list = w_val_dataframe['k_topics'].unique()
    viz_mat = np.zeros((len(w_val_dataframe), len(k_topic_list)))
    
    #Reducing the topic similarity dataframe to only include pairings that are actually similar
    #topic_similarity_dataframe.to_csv('./Document_Data/Plots/raw_similarity.csv')
    topic_similarity_dataframe = topic_similarity_dataframe[topic_similarity_dataframe['same_flag'] == 1]
    
    
    topic_label_list = []
    final_topic_label_list = []
    
    viz_index = 0
    
    pairs_to_remove = []
    
    for k_topics, topic_index, w_value, topic_name in w_val_dataframe.values:
        topic_label_list.append(topic_name)
        
        if [k_topics, topic_index] not in pairs_to_remove:
        
            k_index = np.where(k_topic_list == k_topics)[0][0]
            
            viz_mat[viz_index, k_index] = w_value
            
            #Adding in the topics that are identical
            if k_index != len(k_topic_list) - 1: #Ensuring that if the program is at the last topic it will not run this
                
                #Inputing the current similarity index into this loop
                current_sim_index = topic_index
                
                #Looping over all the different topics remaining in the dataset
                for next_k in k_topic_list[k_index+1:]:
                    
                    #Creating a dataframe that only holds mappings between the two topics
                    similarity_df = topic_similarity_dataframe[topic_similarity_dataframe['k_topics_i'] == k_topics]
                    similarity_df = topic_similarity_dataframe[topic_similarity_dataframe['k_topics_j'] == next_k]
                    
                    if len(similarity_df) == 0: #If the new dataframe is zero (there is no mapping), there is no case to test.
                        break
                    else:
                        new_k_ind = np.where(k_topic_list == next_k)[0][0]
                        new_w_val = 0
                        
                        #This creates a list of topic indices from the new k that are similar to the topics from the previous k                
                        index_list = similarity_df[similarity_df['topic_index_i'] == current_sim_index]['topic_index_j'].values
                        angle_list = similarity_df[similarity_df['topic_index_i'] == current_sim_index]['angle'].values
                        
                        if len(index_list) == 0:
                            break
                        else:
                            new_sim_index = index_list[np.where(angle_list == angle_list.max())[0][0]]
                        
                            #Writes the new w value                    
                            next_topic_data = w_val_dataframe[(w_val_dataframe['k_topics'] == next_k) & (w_val_dataframe['topic_index'] == new_sim_index)]
                            new_w_val = next_topic_data['w_value'].values[0]
                            topic_name = next_topic_data['topic_name'].values[0]
                            viz_mat[viz_index, new_k_ind] = new_w_val
                    
                        #As this has already been added, it should be removed in future runs:
                        pairs_to_remove.append([next_k, new_sim_index])
                    
                        #resets the topic index lists
                        current_sim_index = new_sim_index
                        new_sim_index = 0
                
        final_topic_label_list.append(topic_name)
        
        viz_index += 1
        
    #Updating the plot information to only include topics that have nonzero values in the matrix:
    total_list = [viz_mat[i,:].sum() for i in range(viz_mat.shape[0])]
    
    num_nonzero_topics = 0
    for val in total_list:
        if val >0: 
            num_nonzero_topics += 1
        
    new_viz_mat = np.zeros((num_nonzero_topics, len(k_topic_list)))
    new_topic_label_list = []
    new_final_topic_label_list = []
    
    new_i = 0
    for i in range(viz_mat.shape[0]):
        if total_list[i] >0:
            new_viz_mat[new_i] = viz_mat[i]
            new_topic_label_list.append(topic_label_list[i])
            new_final_topic_label_list.append(final_topic_label_list[i])
            new_i += 1
    
    new_final_topic_label_list.reverse()     
          
            
    fig, ax = plt.subplots(figsize=(len(.5*k_topic_list)+2,int(.5*num_nonzero_topics)+1))
    ax.imshow(new_viz_mat, aspect='auto', cmap = 'nipy_spectral')
    
    #Plotting all the initial topic names
    ax = plt.gca()
    
    ax.set_xticks(np.arange( len(k_topic_list)))
    ax.set_yticks(np.arange( len(new_topic_label_list)))
    
    
    # ... and label them with the respective list entries
    ax.set_xticklabels(k_topic_list)
    ax.set_yticklabels(new_topic_label_list)
    
    ax.set_xticks(np.arange(-.5, len(k_topic_list),1), minor = True)
    ax.set_yticks(np.arange(-.5, len(new_topic_label_list), 1), minor = True)
    
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    #Adding in the final topic labels
    ax2 = ax.twinx()
    
    ax2 = plt.gca()
    
    ax2.set_xticks(np.arange( len(k_topic_list)))
    ax2.set_yticks(np.arange( len(new_final_topic_label_list)))
    
    ax2.set_xticklabels(k_topic_list)
    ax2.set_yticklabels(new_final_topic_label_list)
    
    ax2.set_xticks(np.arange(-.5, len(k_topic_list),1), minor = True)
    ax2.set_yticks(np.arange(-.5, len(new_final_topic_label_list), 1), minor = True)

    # Loop over data dimensions and create text annotations.
    for j in range(len(k_topic_list)):
        norm = sum(new_viz_mat[:, j])
        for i in range(len(new_topic_label_list)):    
            if new_viz_mat[i, j]>0:
                ax.text(j, i, str(round(new_viz_mat[i, j]/norm*100,1))+'%',
                           ha="center", va="center", color="w")
    
    ax.set_title("Topic Scan Through K Topics")
    ax.set_xlabel('k Topics')
    ax.set_ylabel('Initial Top Words in Topic')
    ax2.set_ylabel('Final Top Words in Topic')
    
    fig.tight_layout()

    fig.savefig(plot_save_path+'{}_Topic_K_Trace.pdf'.format(run_name))

    plt.close('all') 
    
  
    
    
    
    
def create_word_image(value_matrix, name_matrix, run_name, report_image_location, k_topics):
    
    fig, ax = plt.subplots(figsize=(2+int(1.15*value_matrix.shape[1]),1+int(.1*value_matrix.shape[0])))
    ax.imshow(value_matrix, aspect='auto', cmap = 'nipy_spectral')
    
    # We want to show all ticks...
    ax = plt.gca()
##    
    ax.set_xticks(np.arange( value_matrix.shape[1]))

    # Loop over data dimensions and create text annotations.
    for i in range(name_matrix.shape[0]):
        for j in range(name_matrix.shape[1]):
            if name_matrix[i, j] != 0:
                ax.text(j, i, name_matrix[i, j],
                           ha="center", va="center", color="w", fontsize = 6)
    
    ax.set_title("Topic Report: {} Topics".format(str(k_topics)))
    ax.set_ylabel('Rank of Word')
    ax.set_xlabel('k Topics')
#    fig.colorbar(im)
    fig.tight_layout()
    plt.savefig(report_image_location+'Topic_Reports/'+run_name+'_Topic_Definitions_k_topics_'+str(k_topics)+'.pdf',dpi = 150)
    
    plt.clf()
    plt.cla()
    plt.close('all')
    
    
def plot_topic_report(run_name, h_filenames, doc_term_filename, plot_save_path, data_save_path, num_terms):
    
    if not os.path.exists(plot_save_path+'Topic_Reports/'):
        os.makedirs(plot_save_path+'Topic_Reports/')
        print('Created folder to save Topic Reports')
    
    
    [doc_term_mat, vocab] = pickle.load(open( doc_term_filename, "rb" ) )   
            
    for p, h_file in enumerate(h_filenames):
    
        (final_weights, H, entropy, invalid_docs) = pickle.load(open(h_file,'rb'))
        num_topics = H.shape[0]
        
        print('Working on topic report for {} topics'.format(str(num_topics)))
                
        value_matrix = np.zeros((num_terms, num_topics))
        name_matrix = np.zeros((num_terms, num_topics), dtype = 'object')
    
        for topic_index in range(num_topics):
            
            top_terms = get_descriptor( vocab, H, topic_index, num_terms )
            top_values = get_top_values( vocab, H, topic_index, num_terms )
            
            row = 0
            for row in range(num_terms):
                value_matrix[row, topic_index] = top_values[row]
                name_matrix[row, topic_index] = top_terms[row]
            
            create_word_image(value_matrix, name_matrix, run_name, plot_save_path, num_topics)
            
            
            
            
            
            
            
def df_to_mat(df, x_col, y_col, z_col):
    
    x = df[x_col].unique()
    x.sort()
    y = df[y_col].unique()
    y.sort()

    
    z_mat = np.zeros((len(y), len(x)))
    
    for p, x_val in enumerate(x):
        for q, y_val in enumerate(y):
            
            z_val = df[(df[x_col] == x_val) & (df[y_col] == y_val)][z_col].values[0]
    
            z_mat[q,p] = z_val
            
    return x, y, z_mat

           
def plot_topic_assignment_scan(plot_folder, data_folder, run_name, h_filenames, dt_filename, angle_list, *args,
                               fig_size = (20,5),
                               num_contours = 50, 
                               cmap_choice = 'viridis',
                               file_type = '.png',
                               **kwargs):
    
    
    angle_threshold_list = []
    num_topic_list = []

    #Holds the entropy of the distribution
    entropy_list = []
    #Holds the mean number of nonzero H values
    mean_nonzero_H_list = []
    #Holds how many documents have no valid angles
    num_invalid_docs_list = []
    
    doc_term_mat, vocab = pickle.load(open(dt_filename, 'rb'))

    num_docs = doc_term_mat.shape[0]
    
    data_filename = data_folder + '{}_topic_assignment_scan_data.pkl'.format(run_name)
    
    try:
        data_df = pickle.load(open(data_filename, 'rb'))
    
    except FileNotFoundError:
            
        
        for angle_threshold in angle_list:
            for h_file in h_filenames:
                            
                final_weights, H, entropy, invalid_docs = assign_topics(angle_threshold, h_file, dt_filename)
                
                num_topic_list.append(H.shape[0])
                angle_threshold_list.append(angle_threshold)
                
                
                entropy_list.append(entropy)
                num_invalid_docs_list.append(len(invalid_docs))
                mean_nonzero_H_list.append(np.nanmean(np.count_nonzero(H, axis = 1)))
                
        data_df = pd.DataFrame({'num_topics': num_topic_list,
                                'angle_threshold': angle_threshold_list,
                                'entropy':entropy_list,
                                'num_invalid':num_invalid_docs_list,
                                'mean_nonzero_H':mean_nonzero_H_list
                                })
        
        pickle.dump(data_df, open(data_filename, 'wb'))
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = fig_size)
    
    for ax, z_col, z_title, z_levels in zip((ax1, ax2, ax3), 
                                            ['entropy', 'num_invalid','mean_nonzero_H'], 
                                            ['Entropy', 'Percent Invalid Docs', 'Order of Mag of Mean Nonzero H Values'],
                                            [[],
                                             [5,10,15,20,25],
                                             [1,1.5,2,2.5,3]]):
        
        x,y,z = df_to_mat(data_df, 'num_topics', 'angle_threshold', z_col)
        
        if z_col == 'mean_nonzero_H':
            z = np.log10(z)
            # c1 = ax.contourf(x, y, z, num_contours, cmap= cmap_choice, locator= ticker.LogLocator())
        elif z_col == 'num_invalid':
            z = z/num_docs*100
        else:
            pass
            
        c1 = ax.contourf(x, y, z, num_contours, cmap= cmap_choice)
        
        c2 = ax.contour(c1, levels=z_levels, colors='white', linestyle = '--', alpha = .7)

            
        cbar = fig.colorbar(c1, ax=ax)
        cbar.add_lines(c2)


        ax.set_xlabel('K Topics')
        ax.set_ylabel('Angle Threshold (Degrees)')
        ax.set_title(z_title)
    
    fig.savefig(plot_folder+'{}_Topic_Assignment_Scan{}'.format(run_name, file_type))

    plt.close('all')     
            
            
            