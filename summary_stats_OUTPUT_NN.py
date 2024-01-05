#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:36:09 2018

@author: baylor
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
from bisect import bisect_left
from scipy import signal
import accl_functions as af
from scipy import stats
import pandas as pd
from scipy import stats
import matplotlib.animation as animation
import time
import new_accl_fns_classes as nacl



##################################################################
plot_single_trial = False
tsne = False
nearest_nabe = False

npzfile = np.load('Train_Test_Data.npz',allow_pickle=True)
training_labels = npzfile['training_labels']
training_data = npzfile['training_data']
validation_set = npzfile['test_set']
validation_labels = npzfile['test_labels']
# all_data_classes=npzfile['all_data_dict'].item()

if plot_single_trial:
    
    single_session = np.load('Train_Test_Data_Single_Video.npz')
    test_set_single = single_session['test_set_single']
    test_labels_single = single_session['test_labels_single']



network_predictions = np.load('Train_Test_Data_Plotting_Test_Set_All.npz',allow_pickle=True)
soft_max_prediction = network_predictions['soft_max_prediction']
logit_diff = network_predictions['logit_diff']
# layer_info = network_predictions['layer1_output']
# training_loss = network_predictions['avg_cost_array']
# validation_loss = network_predictions['avg_cost_validation_array']

#%%

f_accl = ['weardata180206S00A1583.csv','weardata180206S00A1585.csv','weardata180208S00A1573.csv','weardata180308S04A1573.csv',
        'weardata180412S08A1585.csv', 'weardata180511S00A1693.csv','weardata2180511S00A1695.csv','weardata180511S00A1720.csv',
        'weardata180516S00A1736.csv','weardata180523S00A1694.csv','weardata180601S03A1693.csv','weardata180601S03A1695.csv',
        'weardata180601S03A1720.csv','weardata180607S03A1736.csv']

##Define annotation array
f_anno = ['180206S00A1583.csv','180206S00A1585.csv','180208S00A1573.csv', '180308S04A1573.csv', '180412S08A1585.csv', 
        '180511S00A1693.csv','180511S00A1695.csv','180511S00A1720.csv','180516S00A1736.csv','180523S00A1694.csv',
        '180601S03A1693.csv','180601S03A1695.csv','180601S03A1720.csv','180607S03A1736.csv']


###########################################################################
###########################################################################
data_dict = {}

for f,file_name in enumerate(f_accl):
    name = str(file_name[7:])
    key = str(file_name[7:])
    name = nacl.accelerometer_profile()
    name.read_accelerometer_data(file_name)
    name.read_event_data(f_anno[f])
    data_dict[key] = name
   
training_data = []
training_labels = []
test_set = []
test_labels = []
validation_set_dict = {}
validation_labels_dict = {}        
avg_behavior_time_dict = {}
avg_ejac_dict = {}
b = []
test_set_all = list(data_dict.keys())[9:12]
size_each_file = []
#%%
###############################################################################
###############################################################################
def plot_rolling_median(logit_array,ground_truth_array,dim1,dim2,cutoff,rolling_median_window,vid_start,vid_stop,title_name):
    
    a = logit_array[vid_start:vid_stop,dim1] - logit_array[vid_start:vid_stop,dim2]
    #dont care about values less than 0
    a[a<=cutoff] = 0
    a_norm = a/np.max(a)
    #a_median=pd.rolling_median(a_norm,rolling_median_window)
    a_median = pd.Series(a_norm)
    a_median = a_median.rolling(300,min_periods=(1)).median()
    start_real = 0
    start_model = 0
    start_times_model = []
    stop_times_model = []
    start_times_real = []
    stop_times_real = []

    stop_model = 0
    stop_real = 0
    mount_diff_array_model = []
    mount_diff_array_real = []
    

    threshold = 0.001
    Mount = False
    Mount_Real = False
    for i in range(a_median.shape[0]):
      
      
      #ground truth data mount start
      if ground_truth_array[i+vid_start,dim1] == 1 and Mount_Real == False:
        start_real = i
        start_times_real.append(start_real)
        Mount_Real = True
      
         #ground truth data mount stop
      elif i > start_real and Mount_Real and ground_truth_array[vid_start + i + 1,dim1] == 0:
        stop_real = i  
        stop_times_real.append(stop_real)
        mount_diff_array_real.append(stop_real - start_real)
        Mount_Real = False
        
        
      #NN data mount start
      elif a_median[i] > threshold and Mount == False:
        start_model = i
        
        Mount = True
        
      #NN data mount stop
      elif i > start_model and Mount and a_median[i] < threshold:
        stop_model=i
        if stop_model - start_model > 250:
          start_times_model.append(start_model)
          stop_times_model.append(stop_model)
          mount_diff_array_model.append(stop_model - start_model)
        Mount=False
    

    plt.figure()
    plt.ylim((-1.2,1.2))
    plt.plot(a_median,'b',label='Output from Network')
    plt.plot(ground_truth_array[vid_start:vid_stop,dim1], 'r',label='Ground Truth')
    plt.title('Test Set %s' %title_name)
    plt.xlabel('Time (100ms)')
    plt.ylabel('Logit Difference')
    plt.legend(loc=4)
    
    return a_median, mount_diff_array_model,start_times_model,stop_times_model,mount_diff_array_real,start_times_real,stop_times_real, a
###############################################################################
###############################################################################
def calculate_IMI(mounting_info_array):
  """Calculates the inter-mount Interval of all of the sessions"""
  
  imi = []
  for i in range(np.size(mounting_info_array) - 1):
    imi.append(mounting_info_array[i + 1] - mounting_info_array[i])
  return imi
###############################################################################
###############################################################################
def mean_sex(sex_array):
  
    return np.mean(sex_array)
###############################################################################
###############################################################################
name_array = []

for key,value in data_dict.items():
    
    if key in test_set_all:    
      avg_behavior_time_dict[key],avg_ejac_dict[key] = af.get_avg_mount_time(value.indices,value.event_data)
      print(key)
      print(np.shape(value.x_data))
      size_each_file.append(len(value.x_data))
      name_array.append(key[1:15])

for key, value in avg_behavior_time_dict.items():

        b.append(np.mean(value))

total_mean_behavior = np.mean(b)
total_behavior_SEM = stats.sem(b)/100
###############################################################################
###############################################################################

start_sess = 0
filtered_prediction = []
mount_array_network_prediction = []
start_times_nn = []
stop_times_nn = []
mount_array_network_rl = []
start_times_rl = []
stop_times_rl = []
t = 0

""" description of variables
filtered_pred = normalized output of network """
for vid in size_each_file:
  plot_title=name_array[t]
  stop_sess=vid+start_sess+1
  
  filtered_prediction_var, mount_array_network_prediction_var, \
      start_times_nn_var,stop_times_nn_var, \
      mount_array_network_rl_var,start_times_rl_var,stop_times_rl_var, \
          tree \
          =plot_rolling_median(logit_diff,validation_labels,0,1,0,300,start_sess,stop_sess,plot_title)
  
  filtered_prediction.append(filtered_prediction_var)
  mount_array_network_prediction.append(mount_array_network_prediction_var)
  start_times_nn.append(start_times_nn_var)
  stop_times_nn.append(stop_times_nn_var)
  mount_array_network_rl.append(mount_array_network_rl_var)
  start_times_rl.append(start_times_rl_var)
  stop_times_rl.append(stop_times_rl_var)
  
  
  start_sess=stop_sess-1
  t+=1

#%%Calculate Stats for NN output and groundtruth data
mean_sex_array_nn = []
mean_sex_array_rl = []
imi_real = []
imi_nn = []
imi_real_std = []
imi_nn_std = []
std_sex_array_nn = []
std_sex_array_rl = []
mount_count_array_nn = []
mount_count_array_rl = []

for i in range(3):
  mean_sex_nn = mean_sex(mount_array_network_prediction[i])
  mean_sex_rl = mean_sex(mount_array_network_rl[i]) 
  imi_mean_nn = np.mean(calculate_IMI(start_times_nn[i]))
  imi_mean_rl = np.mean(calculate_IMI(start_times_rl[i]))
  
  std_sex_nn = np.std(mount_array_network_prediction[i])
  std_sex_rl = np.std(mount_array_network_rl[i])
  std_imi_nn = np.std(calculate_IMI(start_times_nn[i]))
  std_imi_rl = np.std(calculate_IMI(start_times_rl[i]))
  
  mount_count_nn = np.size(mount_array_network_prediction[i])
  mount_count_rl = np.size(mount_array_network_rl[i])
  
  #mean sex array is average mount time
  mean_sex_array_nn.append(mean_sex_nn)
  mean_sex_array_rl.append(mean_sex_rl)
  #imi is inter mount interval
  imi_real.append(imi_mean_rl)
  imi_nn.append(imi_mean_nn)
  imi_real_std.append(std_imi_rl)
  imi_nn_std.append(std_imi_nn)
  
  std_sex_array_nn.append(std_sex_nn)
  std_sex_array_rl.append(std_sex_rl)
  
  mount_count_array_nn.append(mount_count_nn)
  mount_count_array_rl.append(mount_count_rl)

#convert from milliseconds to seconds
mean_sex_array_nn = [x / 100 for x in mean_sex_array_nn]
mean_sex_array_rl = [x / 100 for x in mean_sex_array_rl]
imi_real = [x / 100 for x in imi_real]
imi_nn = [x / 100 for x in imi_nn]
imi_real_std = [x / 100 for x in imi_real_std]
imi_nn_std = [x / 100 for x in imi_nn_std]
std_sex_array_nn = [x / 100 for x in std_sex_array_nn]
std_sex_array_rl = [x / 100 for x in std_sex_array_rl]



# outfile='Filtered_Data'
# f0 = filtered_prediction[0]
# f1 = filtered_prediction[1]
# f2 = filtered_prediction[2]
# f3 = filtered_prediction[3]
# f4 = filtered_prediction[4]

# np.savez(outfile,f0 = f0,f1 = f1)
# #
#
#%%

def plot_mean(control_array, test_array):
    SAL = control_array

    DCZ=test_array 


    SAL_mean=np.mean(SAL)
    SAL_std=np.std(SAL)
    DCZ_mean=np.mean(DCZ)
    DCZ_std=np.std(DCZ)

    t_stat_lateral,p_val=stats.ttest_ind(SAL,DCZ)


    # Create lists for the plot
    materials = ['Ground Truth', 'Neural Network',]
    x_pos = np.arange(len(materials))
    CTEs = [SAL_mean, DCZ_mean]
    error = [SAL_std, DCZ_std]
    w = 0.4   # bar width
    
    
    #'seagreen'
    #dodgerblue
    #marigold
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, 
            align='center', 
            ecolor='black', 
            color=['mintcream','firebrick'],
            capsize=10,
            alpha = 1,
            edgecolor='black',
            width=w)

    i=0
    t=0
    s=.98
    # 
    for i in range(len(SAL)):
        ax.scatter(t, SAL[i], color='black')
        ax.scatter(s, DCZ[i], color='black')
        ax.plot((t,s),(SAL[i],DCZ[i]), color='black')


    ax.set_ylabel('Total Number of Mounts')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(materials)
    ax.set_title('Mounts Bouts/Session Comparison')

    ax.yaxis.grid(False)


    # Save the figure and show
    plt.tight_layout()
    plt.ylim([0,30])
    plt.savefig('bar_plot_with_error_bars.png')
    plt.show()

    print(p_val)
    
plot_mean(imi_real, imi_nn)
plot_mean(mean_sex_array_rl, mean_sex_array_nn)
plot_mean(std_sex_array_nn, std_sex_array_rl)
plot_mean(mount_count_array_rl, mount_count_array_nn)


#%%
def confusion_calc(gt_array, pred_array):
    '''Takes ground truth array and predictions from the network and determines the confusion matrix.
    Using a 0.5 cut off for class prediction. Currently the pred array is using the softmax output of the 
    network. Data has not been filtered or processed at this step'''

    #number of test sessions
    nTstSz = np.size(gt_array)
    confusion_dict = dict()
    confusion_matrix_all = [[0,0,0,0],[0,0,0,0]]
    GtNoSex_total = 0
    GtSex_total = 0
    #loop through each data set
    
    for i in range(nTstSz):
        arPred = np.asarray(pred_array[i])
        arGT = np.asarray(gt_array[i])
        #0 axis contains all the predictions [0,1] or [1,0]
        nPrediction = np.size(arGT,0)
        print(nPrediction)
        #true/false positive counts
        #these values are place holders for confusion matrix
        GtNoSex = 0
        GtSex = 0
        #initialize confusion matrix
        confusion_matrix = [[0,0,0,0],[0,0,0,0]]

        for t in range(nPrediction):

            # if softmax is greater 0.5, classify as sex
            # true positive
            if arPred[t][0] > 0.5 and arGT[t][0] == 1:
                #individual 
                confusion_matrix[0][0] = confusion_matrix[0][0] + 1
                confusion_matrix[1][0] = confusion_matrix[1][0] + 1
                #group
                confusion_matrix_all[0][0] = confusion_matrix_all[0][0] + 1
                confusion_matrix_all[1][0] = confusion_matrix_all[1][0] + 1
                
                GtSex = GtSex + 1
                GtSex_total = GtSex_total + 1
            
            # false negative
            if arPred[t][0] <= 0.5 and arGT[t][0] == 1:
                #individual
                confusion_matrix[0][1] = confusion_matrix[0][1] + 1
                confusion_matrix[1][1] = confusion_matrix[1][1] + 1
                #group
                confusion_matrix_all[0][1] = confusion_matrix_all[0][1] + 1
                confusion_matrix_all[1][1] = confusion_matrix_all[1][1] + 1
                
                GtSex = GtSex + 1
                GtSex_total = GtSex_total + 1
            
            #true negative
            elif arPred[t][0] <= 0.5 and arGT[t][0] == 0:
                #individual
                confusion_matrix[0][2] = confusion_matrix[0][2] + 1
                confusion_matrix[1][2] = confusion_matrix[1][2] + 1
                #group
                confusion_matrix_all[0][2] = confusion_matrix_all[0][2] + 1
                confusion_matrix_all[1][2] = confusion_matrix_all[1][2] + 1
                
                GtNoSex = GtNoSex + 1
                GtNoSex_total = GtNoSex_total + 1
                
            #false Positive
            elif arPred[t][0] > 0.5 and arGT[t][0] == 0:
                #individual
                confusion_matrix[0][3] = confusion_matrix[0][3] + 1
                confusion_matrix[1][3] = confusion_matrix[1][3] + 1
                #group
                confusion_matrix_all[0][3] = confusion_matrix_all[0][3] + 1
                confusion_matrix_all[1][3] = confusion_matrix_all[1][3] + 1
                
                GtNoSex = GtNoSex + 1
                GtNoSex_total = GtNoSex_total + 1
        
        print(GtSex)
        print(GtNoSex)
        # TP rate    
        confusion_matrix[1][0] = confusion_matrix[1][0] / GtSex
        #FP rate
        confusion_matrix[1][3] = confusion_matrix[1][3] / GtNoSex
        #FN rate
        confusion_matrix[1][1] = confusion_matrix[1][1] / GtSex
        #TN rate
        confusion_matrix[1][2] = confusion_matrix[1][2] / GtNoSex
           
        confusion_dict[i] = confusion_matrix
        
    # TP rate    
    confusion_matrix_all[1][0] = confusion_matrix_all[1][0] / GtSex_total
    #FP rate
    confusion_matrix_all[1][3] = confusion_matrix_all[1][3] / GtNoSex_total
    #FN rate
    confusion_matrix_all[1][1] = confusion_matrix_all[1][1] / GtSex_total
    #TN rate
    confusion_matrix_all[1][2] = confusion_matrix_all[1][2] / GtNoSex_total

        
        
        
        
    return confusion_dict, confusion_matrix_all

walk, walkhard = confusion_calc(validation_labels,soft_max_prediction)    





