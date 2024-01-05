#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 15:05:52 2018

@author: baylor
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import accl_functions_OneHot as af
# from sklearn.manifold import *
# from sklearn import *
import accl_functions as af
# from sklearn.neighbors import *
from scipy import stats

import matplotlib.animation as animation
import time


##Load all data from network##
"""The validation labels are stored as the following
    0-no sex
    1-sexual behavior
    2-sexual behavior attempt
    3-sniffing
    4-grooming"""
##################################################################
    
##################################################################
plot_single_trial=False
tsne=False
nearest_nabe=False

npzfile = np.load('Train_Test_Data.npz')
training_data=npzfile['training_data']
training_labels=npzfile['training_labels']
validation_set=npzfile['test_set']
validation_labels=npzfile['test_labels']
#all_data_classes=npzfile['all_data_dict'].item()

if plot_single_trial:
    
    single_session=np.load('Train_Test_Data_Single_Video.npz')
    test_set_single=single_session['test_set_single']
    test_labels_single=single_session['test_labels_single']



network_predictions=np.load('Train_Test_Data_Plotting.npz')
soft_max_prediction=network_predictions['soft_max_prediction']
logit_diff=network_predictions['logit_diff']
layer_3_network=network_predictions['tsne_layer3']
training_loss=network_predictions['avg_cost_array']
validation_loss=network_predictions['avg_cost_validation_array']




##################################################################

##################################################################
def plot_behaviors(x_data , y_data , z_data , indices , event_data, normalize_values):
    """Plots behaviors returned from accl_extract"""
    
    if normalize_values:
    #normalize accl values
        new_x = (x_data-np.mean(x_data))/np.var(x_data)
        new_y = (y_data-np.mean(y_data))/np.var(y_data)
        new_z = (z_data-np.mean(z_data))/np.var(z_data)
    if not normalize_values:
        new_x = x_data
        new_y = y_data
        new_z = z_data
    
    fig = plt.figure()
    plt.plot(new_x, '#1A8925') #green
    plt.plot(new_y, '#891A7D') #purple
    plt.plot(new_z, '#02BACF') #light blue
#    plt.plot(indices, np.zeros_like(indices), 'o' , color = 'r')
#    for x in range (0,len(indices)):
#        if event_data[x] == 'Ejaculation':
#            plt.plot(indices[x] , 0.0 , 'o' , color = 'k')
#        if event_data[x] == 'MountIn': 
#            plt.plot(indices[x] , 0.0 , 'o' , color = 'r')
#        if event_data[x] == 'MountOut': 
#            plt.plot(indices[x] , 0.0 , 'o' , color = 'g')
#        if event_data[x] == 'Intromission': 
#            plt.plot(indices[x] , 0.0 , 'o' , color = 'b')
#        if event_data[x] == 'MountAttempt': 
#            plt.plot(indices[x] , 0.0 , 'o' , color = 'm')

    ax = fig.add_subplot(111)
    
    for x in range (0,len(indices)):
       if event_data[x] == 'MountIn': ax.annotate('MI', xy=(indices[x], 0.001), xycoords='data' , color = 'r')
       if event_data[x] == 'MountOut': ax.annotate('MO', xy=(indices[x], 0.001), xycoords='data' , color = 'g')
       if event_data[x] == 'Intromission': ax.annotate('I', xy=(indices[x], 0.001), xycoords='data' , color = 'b')
       if event_data[x] == 'Ejaculation': ax.annotate('E', xy=(indices[x], 0.001), xycoords='data' , color = 'c')    
       if event_data[x] == 'MountAttempt': ax.annotate('A', xy=(indices[x], 0.001), xycoords='data' , color = 'm')
    plt.show()
##################################################################

##################################################################    
def find_min_vals(val_cost_array):
    """Find the min value of the cost function from training and returns the 
    index. This function is used to determine which weight-epoch to use
    for test sets"""
    
    vals=np.nonzero(val_cost_array==np.min(val_cost_array))
    return vals

##################################################################
##################################################################
#def time_points_for_behaviors(logit_diff_array):
#    """Returns an array if the start or stop times of the beahviors,
#    as predicted by the network"""
#    
#    
#    #cut off time is the average length of the behavior in question
#    cut_off_time=15
#    if val >0.2:
#        if np.median(logit_diff_array[val:val+cut_off_time])>0.2:
#    
#    
#    
#    return time_point_array
##################################################################

##################################################################

# def plot_logit_diff(logit_array,ground_truth_array,dim1,dim2):
#     plt.figure()
#     ax1=plt.subplot(211)
#     plt.plot((logit_diff[:,dim1]-logit_diff[:,dim2]),'g')
#     plt.title('Logit Difference \n 500 Epochs')
#     plt.xlabel('time')
#     plt.ylabel('activation_model')



# def plot_multi_class_prediction(softmax_output_array, number_of_classes,name_array):
#  """Name array has the list that corresponds to the softmax order. Make sure it
# is filled correctly"""
# fig = plt.figure()
#   for i in range(number_of_classes):
#     ax1 = fig.add_subplot(221)
#     ax2 = fig.add_subplot(222)
#     ax3 = fig.add_subplot(223)
#     ax4 = fig.add_subplot(224)
#     ax1.title.set_text('Mount Attempt')
#     ax2.title.set_text('Intromissions')
#     ax3.title.set_text('Sniffing')
#     ax4.title.set_text('Grooming')


  
##################################################################
"""Plot rolling median, uses pandas. Make sure you select dim
and dim2 correctly!"""
##################################################################
def plot_rolling_median(logit_array,ground_truth_array,dim1,dim2,cutoff,rolling_median_window):
    
    plt.figure()
    a=logit_array[:,dim1]-logit_array[:,dim2]
    #dont care about values less than 0
    a[a<=cutoff]=0
    a_norm=a/np.max(a)
    a_median=pd.core.window.rolling.Rolling.median((a_norm,rolling_median_window))
    start_real=0
    start_model=0
    start_times_model=[]
    stop_times_model=[]
    start_times_real=[]
    stop_times_real=[]

    stop_model=0
    stop_real=0
    mount_diff_array_model=[]
    mount_diff_array_real=[]
    
    idx_check=0
    threshold=0.001
    rho=20
    Mount=False
    Mount_Real=False
    for i in range(a_median.shape[0]):
      
      
      #ground truth data mount start
      if ground_truth_array[i,dim2]==1 and Mount_Real==False:
        start_real=i
        start_times_real.append(start_real)
        Mount_Real=True
      
         #ground truth data mount stop
      elif i > start_real and Mount_Real and ground_truth_array[i+1,dim2]==0:
        stop_real=i  
        stop_times_real.append(stop_real)
        mount_diff_array_real.append(stop_real-start_real)
        Mount_Real=False
        
        
      #NN data mount start
      elif a_median[i]>threshold and Mount==False:
        start_model=i
        
        Mount=True
        
      #NN data mount stop
      elif i > start_model and Mount and a_median[i]<threshold:
        stop_model=i
        if stop_model-start_model>250:
          start_times_model.append(start_model)
          stop_times_model.append(stop_model)
          mount_diff_array_model.append(stop_model-start_model)
        Mount=False
         
    plt.figure()
    plt.plot(a_median)
    plt.plot(ground_truth_array[:,dim1], 'r')
    return a_median, mount_diff_array_model,start_times_model,stop_times_model,mount_diff_array_real,start_times_real,stop_times_real

 
filtered_prediction,mount_array_network_prediction,start_times_nn,stop_times_nn,mount_array_network_rl,start_times_rl,stop_times_rl=plot_rolling_median(logit_diff,validation_labels,0,1,0,300)


def calculate_IMI(mounting_info_array):
  imi=[]
  for i in range(np.size(mounting_info_array)-1):
    imi.append(mounting_info_array[i+1]-mounting_info_array[i])
  return imi
imi_mean_nn=np.mean(calculate_IMI(start_times_nn))
imi_mean_rl=np.mean(calculate_IMI(start_times_rl))

##################################################3
#Summary Statistics######
##################################################3


def get_avg_mount_time(behavior_indicies,event_data):
   event_index=behavior_indicies #array with timepoints for annotations
   final_stop='Ejaculation'
   start=[]
   stop=[]
   idx_check=0
   mount_diff_array=[]
   ejac_time=0



   for counter,value in enumerate(event_data):
       
       if value=='MountIn':
           idx_check=+1
           Mount=True
           start_behavior=event_index[counter]
           start.append(start_behavior)


       if value=='MountOut' or value==final_stop:
           if len(start)==0:
               continue
           #successful mounts have a label of 1   
           elif event_index[counter] > start[idx_check-1] and Mount:
               stop_behavior=event_index[counter]
               if value=='MountOut':
                   mount_diff_array.append(stop_behavior-start_behavior)
               elif value==final_stop:
                   ejac_time=stop_behavior-start_behavior   
               Mount=False
               
               
   return mount_diff_array, ejac_time


def find_threshold(data_array):
  """Finds the cutoff values for our network"""


        
##    avg_behavior_time_dict[key],avg_ejac_dict[key]=af.get_avg_mount_time(value.indices,value.event_data)
##
##for key, value in avg_behavior_time_dict.items():
##
##        b.append(np.mean(value))
#
##total_mean_behavior=np.mean(b)
##total_behavior_SEM=stats.sem(b)/100
#
#
#
#list(avg_behavior_time_dict.values())
#for key, value in avg_behavior_time_dict.items():
#        a.append(value)
#        
#a=[]
#for key, value in avg_behavior_time_dict.items():
#
#        a.append(value)
#
#asum=np.sum(a,2)
#a[1]
#np.sum(a[1])
#a[1].shape
#a[1].size
#np.size(a[1])
#for key, value in avg_behavior_time_dict.items():
#
#        a.append(np.mean(value))
#
#b=[]
#for key, value in avg_behavior_time_dict.items():
#
#        b.append(np.mean(value))
#
#np.mean(b)
#1685/100
#from scipy import stats
#stats.sem(b)
#stats.sem(b)/100

#ax2=plt.subplot(212,sharex=ax1)
##plt.subplot(3, 1, 3)
##plot mount behaior
#plt.plot(validation_labels[:,0], 'r')
##plot intromission data
#
##plt.plot(a.intromission_lables_gt[:,0], 'r')
#plt.xlabel('time')
#plt.ylabel('activation_ground_truth')
#plt.show()
#
#plt.figure()
#a=logit_diff[:,0]-logit_diff[:,1]
##dont care about values less than 0
#a[a<=-5]=0
#a_norm=a/np.max(a)
#plt.plot(a_norm)
#plt.plot(validation_labels[:,1], 'r')
#plt.title('Normalized Logit Difference')
#plt.xlabel('time (ms)')
#plt.ylabel('activation_model')


############################################################
#tSNE 
############################################################
def find_elements(array,element,element2):
    """Finds indicies of an array that match search criteria
    as determined by values of element and element 2. There is a 
    third class, which was created to deal with all of the other entries outside
    of the two primary search terms"""
    
    
    i=0
    index_for_criteria=[]
    contra=[]
    wtf=[]
    for i,t in enumerate(array):
        if np.array_equal(t,element):
            index_for_criteria.append(i)
        if np.array_equal(t,element2):
            contra.append(i)
        if not np.array_equal(t,element) and not np.array_equal(t,element2):
            wtf.append(i)
    print(i)
    return index_for_criteria,contra,wtf



##########################################################
def remove_0_weights(data_array):
  """Removes inactive nodes""" 
  
  reduced_weight_vector=[]
  t=0
  for i in range(np.int(data_array.shape[0])):
    #if a row has not been used during training, leave it out
    if np.count_nonzero(data_array[i,:]) != 0:
      reduced_weight_vector.append(data_array[i,:])
      t+=1
  print((i,t))
  return reduced_weight_vector


##########################################################
def vis_layer_tsne(data_array,class1,class2):
    """Runs and plots tsne on data_array. Class 1 and Class 2
    are used to differentiate the different label types
    from our binary NN classifier"""
    
    
    hidden_layer=data_array

    tsne = manifold.TSNE(n_components=2, perplexity=30,n_iter=10000,init='pca')
    
    X_tsne = tsne.fit_transform(hidden_layer)
#
#    #nointromission subset of tsne
    x_no_intromission=X_tsne[class1]    
#    #intromission
    x_intromission=X_tsne[class2]
    
    #first axis of 2-d tsne
    X0_intromission=x_intromission[:,0]
    X0_no_intromission=x_no_intromission[:,0]

    #2nd axis of 2-d tsne    
    X1_intromission=x_intromission[:,1]
    X1_no_intromission=x_no_intromission[:,1]

    plt.figure()
    ax2=plt.subplot(211)
    ax1=plt.subplot(211)
    ax1.scatter(X0_intromission, X1_intromission, c="g")
    ax1.scatter(X0_no_intromission,X1_no_intromission , c="r")

#    ax2.scatter(X_tsne[:246607,0],X_tsne[:246607,1],c='g')
#    ax2.scatter(X_tsne[246607:,0],X_tsne[246607:,1],c='r')
    plt.show()
#    return X0_intromission,X0_no_intromission,X1_intromission,X1_no_intromission
#####################################################
#define vectors for tsne
#####################################################
if tsne:
  shuffle_layer3,shuffle_labels =af.shuffle_data_tsne(layer_3_network,validation_labels)
  
  #choose a random sample of shuffled data, the more data the longer tsne will take
  sub_sample_layer3=shuffle_layer3[1:10001]
  sub_sample_layer3_labels=shuffle_labels[1:10001]
  
  #be careful that you use the correct labels
  no_intromission_label=np.array([0.,1.])
  intromission_label=np.array([1.,0.])

  #value for upper limit comes from size of single vid, calculated at start
#  layer_3_network_first_vid=layer_3_network[:,:246607]

  #find indicies in validation label, used for extracting subspaces, see below
  no_intromission,intromission,wtf1=find_elements(sub_sample_layer3_labels,no_intromission_label,intromission_label)

  # find subspaces 
#  just_sex=layer_3_network_first_vid[:,intromission]
#  no_sex=layer_3_network_first_vid[:,no_intromission]

#  just_sex_non_zero_weights=remove_0_weights(just_sex)

#  first_vid_dim=np.ones((1,246607))
  
  #plot tsne results
  vis_layer_tsne(sub_sample_layer3,no_intromission,intromission)

#########################################################################
#Nearest Neighbor for tsne validation and cross-checking
#########################################################################

def n_nabe(data_array,label_array,number_of_nabes):


  X=np.transpose(data_array)

  diff_knabes_test=[]
  tree = KDTree(X, leaf_size=5000)
              
  #comparison_vector=np.zeros(number_of_nabes)
  data_range=X.shape[0]
  for t in range(data_range):
    dist, ind = tree.query([X[t]], k=number_of_nabes)
    #get the labels of the k-nearest nabes
    #comparison_vector=ind
    #find label of current trial
    sample_label=label_array[t]
    p=0
    for i in range(number_of_nabes):
      index_for_validation=ind[:,i]
      nearest_nabe_label=label_array[index_for_validation,:]      
      if np.array_equal(nearest_nabe_label,sample_label):
        
        p+=1
    
    diff_knabes_test[t]=p/100
    
    
  return diff_knabes_test
if nearest_nabe:
  
  nabe_test=n_nabe(layer_3_network,validation_labels,100)