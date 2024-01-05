#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:18:44 2023

@author: baylor

Naive Classifier for thesis
Predicts outcomes from training set and finds theoretical perfomance levels. 
This algo is run on the test set to determine whether or not the accuracy 
levels diverge significicantly from what has been observed


"""
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
from bisect import bisect_left
from scipy import signal
# import accl_functions as af
from scipy import stats
import pandas as pd
from scipy import stats
import matplotlib.animation as animation
import time
# import new_accl_fns_classes as nacl
import random
import copy

##################################################################


# npzfile = np.load('Train_Test_Data_9_12_segmented.npz',allow_pickle=True)
# validation_set = npzfile['test_set']
# validation_labels = npzfile['test_labels']



# network_predictions = np.load('Train_Test_Data_Plotting_9_12_segmented.npz',allow_pickle=True)
# soft_max_prediction = network_predictions['soft_max_prediction']
# logit_diff = network_predictions['logit_diff']


npzfile = np.load('Train_Test_Data_9_12_segmented.npz',allow_pickle=True)
validation_set = npzfile['test_set']
validation_labels = npzfile['test_labels']



network_predictions = np.load('Train_Test_Data_Plotting.npz',allow_pickle=True)
soft_max_prediction = network_predictions['soft_max_prediction']
logit_diff = network_predictions['logit_diff']


def recall_calc(true_positive, false_negative):
    rec_array = []
    for x in range(len(true_positive)):
        rec_array.append(true_positive[x]/(true_positive[x]+false_negative[x]))
    
    mean_rec = np.mean(rec_array)
    std_rec = np.std(rec_array)
    
    return mean_rec, std_rec, rec_array
    

def precision_calc(true_positive, false_positive):
    prec_array = []
    for x in range(len(true_positive)):
        prec_array.append(true_positive[x]/(true_positive[x]+false_positive[x]))
    mean_prec = np.mean(prec_array)
    std_prec = np.std(prec_array)
    return mean_prec, std_prec, prec_array

def fl_calc(recall_val, prec_val):
    f1_val = 2 * ((recall_val * prec_val)/(recall_val + prec_val))
    
    return f1_val

def confusion_calc_shuffled(gt_array, pred_array):
    '''Takes ground truth array and predictions from the network and determines the confusion matrix.
    Using a 0.5 cut off for class prediction. Currently the pred array is using the softmax output of the 
    network. Data has not been filtered or processed at this step'''

    #number of test sessions
    #number of files in test set
    nTstSz = np.size(gt_array)

    #loop through each data set
    ar_true_positive = [] 
    ar_true_negative = []
    ar_false_negative = []
    ar_false_positive = []
    
    n_true_positive_mean = 0 
    n_true_negative_mean = 0
    n_false_negative_mean = 0
    n_false_positive_mean = 0
    
    TP_count = [] 
    FN_count = []
    FP_count = []
    TN_count = []
    for num_iters in range(1000):


        TP_Rate = 0
        FP_Rate = 0
        TN_Rate = 0
        FN_Rate = 0
        nTP = 0
        nTN = 0
        nFN = 0
        nFP = 0
        GtNoSex_total = 0
        GtSex_total = 0
        i = 0
        while i < 3:
            
            #prediction array for test 1 test set from test set array
            arPred = copy.deepcopy(np.asarray(pred_array[i]))
            #GT array for one gt set from ground truth array
            arGT = copy.deepcopy(np.asarray(gt_array[i]))
            
            #shuffle the groundtruth array
            ar_Shuffled_GT = copy.deepcopy(arGT)
            random.shuffle(ar_Shuffled_GT)
            
            nPrediction = np.size(arGT,0)
            #total gt_sex and no sex counts for calculating rates

        
            for t in range(nPrediction):
    
                # if softmax is greater 0.5, classify as sex
                # true positive
                if arPred[t][0] > 0.5 and ar_Shuffled_GT[t][0] == 1:
                    #individual 
                    nTP = nTP + 1
                    GtSex_total = GtSex_total + 1
                
                # false negative
                elif arPred[t][0] <= 0.5 and ar_Shuffled_GT[t][0] == 1:
                    #individual
                    nFN = nFN + 1
                    GtSex_total = GtSex_total + 1
                
                #true negative
                elif arPred[t][0] <= 0.5 and ar_Shuffled_GT[t][0] == 0:
                    #individual
                    nTN = nTN + 1
                    GtNoSex_total = GtNoSex_total + 1
                    
                #false Positive
                elif arPred[t][0] > 0.5 and ar_Shuffled_GT[t][0] == 0:
                    #individual
                    nFP = nFP + 1
                    GtNoSex_total = GtNoSex_total + 1
            i = i +1        
 
        if GtSex_total == 0:
            TP_Rate = 0
            FN_Rate = 0
        else:
            # TP rate    
            TP_Rate = nTP / GtSex_total
            #FN rate
            FN_Rate = nFN / GtSex_total
            
        
        #FP rate
        FP_Rate = nFP / GtNoSex_total
        #TN rate
        TN_Rate = nTN / GtNoSex_total
        
        #counts
        TP_count.append(nTP) 
        FN_count.append(nFN)
        FP_count.append(nFP)
        TN_count.append(nTN)
        
        #rate array
        ar_true_positive.append(TP_Rate)
        ar_true_negative.append(TN_Rate)
        ar_false_negative.append(FN_Rate)
        ar_false_positive.append(FP_Rate)
    #mean of rate
    n_true_positive_mean = np.mean(ar_true_positive) 
    n_true_negative_mean = np.mean(ar_true_negative)
    n_false_negative_mean = np.mean(ar_false_negative)
    n_false_positive_mean = np.mean(ar_false_positive)
        
    return TP_count, TN_count, FN_count, FP_count, n_true_positive_mean, n_true_negative_mean, n_false_negative_mean,n_false_positive_mean

def confusion_calc_lazy_model(gt_array, pred_array):
    '''Takes ground truth array and predictions from the network and determines the confusion matrix.
    Using a 0.5 cut off for class prediction. Currently the pred array is using the softmax output of the 
    network. Data has not been filtered or processed at this step'''

    #number of test sessions
    #number of files in test set
 

    #loop through each data set
    ar_true_positive = [] 
    ar_true_negative = []
    ar_false_negative = []
    ar_false_positive = []
        
    n_true_positive_mean = 0 
    n_true_negative_mean = 0
    n_false_negative_mean = 0
    n_false_positive_mean = 0
    
    TP_count = [] 
    FN_count = []
    FP_count = []
    TN_count = []
        
    
    for num_iters in range(1000):


        TP_Rate = 0
        FP_Rate = 0
        TN_Rate = 0
        FN_Rate = 0
        nTP = 0
        nTN = 0
        nFN = 0
        nFP = 0
        GtNoSex_total = 0
        GtSex_total = 0
        i = 0
        while i < 3:
            
            #GT array for one gt set from ground truth array
            arGT = copy.deepcopy(np.asarray(gt_array[i]))
            
            nPrediction = np.size(arGT,0)
            #total gt_sex and no sex counts for calculating rates

        
            for t in range(nPrediction):
                prediction_choice = random.choices([1,0], [1,9], k=1)
                # prediction_choice = [0]
                # if softmax is greater 0.5, classify as sex
                # true positive
                
                if prediction_choice[0] > 0.5 and arGT[t][0] == 1:
                    #individual 
                    nTP = nTP + 1
                    GtSex_total = GtSex_total + 1
                
                # false negative
                elif prediction_choice[0] <= 0.5 and arGT[t][0] == 1:
                    #individual
                    nFN = nFN + 1
                    GtSex_total = GtSex_total + 1
                
                #true negative
                elif prediction_choice[0] <= 0.5 and arGT[t][0] == 0:
                    #individual
                    nTN = nTN + 1
                    GtNoSex_total = GtNoSex_total + 1
                    
                #false Positive
                elif prediction_choice[0] > 0.5 and arGT[t][0] == 0:
                    #individual
                    nFP = nFP + 1
                    GtNoSex_total = GtNoSex_total + 1
            i = i +1        
 
        # TP rate    
        TP_Rate = nTP / GtSex_total
        #FN rate
        FN_Rate = nFN / GtSex_total
        #FP rate
        FP_Rate = nFP / GtNoSex_total
        #TN rate
        TN_Rate = nTN / GtNoSex_total
        
        TP_count.append(nTP) 
        FN_count.append(nFN)
        FP_count.append(nFP)
        TN_count.append(nTN)
        
        
        
        
        ar_true_positive.append(TP_Rate)
        ar_true_negative.append(TN_Rate)
        ar_false_negative.append(FN_Rate)
        ar_false_positive.append(FP_Rate)
        
    n_true_positive_mean = np.mean(ar_true_positive) 
    n_true_negative_mean = np.mean(ar_true_negative)
    n_false_negative_mean = np.mean(ar_false_negative)
    n_false_positive_mean = np.mean(ar_false_positive)
    
    return TP_count, TN_count, FN_count, FP_count, n_true_positive_mean, n_true_negative_mean, n_false_negative_mean,n_false_positive_mean


def confusion_calc(gt_array, pred_array):
    '''Takes ground truth array and predictions from the network and determines the confusion matrix.
    Using a 0.5 cut off for class prediction. Currently the pred array is using the softmax output of the 
    network. Data has not been filtered or processed at this step'''
    TP = []
    TN = []
    FP = []
    FN = []
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
        TP.append(confusion_matrix[0][0])
        confusion_matrix[1][0] = confusion_matrix[1][0] / GtSex
        #FP rate
        FP.append(confusion_matrix[0][3])
        confusion_matrix[1][3] = confusion_matrix[1][3] / GtNoSex
        #FN rate
        FN.append(confusion_matrix[0][1])
        confusion_matrix[1][1] = confusion_matrix[1][1] / GtSex
        #TN rate
        TN.append(confusion_matrix[0][2])
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

    return confusion_dict, confusion_matrix_all, TP, TN, FP, FN
    
#%%
TP_shuff, TN_shuff, FN_shuff, FP_shuff,TP_shuff_mean, TN_shuff_mean, FN_shuff_mean, FP_shuff_mean  = confusion_calc_shuffled(validation_labels, soft_max_prediction)

TP_lazy, TN_lazy, FN_lazy, FP_lazy, TP_lazy_mean, TN_lazy_mean, FN_lazy_mean, FP_lazy_mean = confusion_calc_lazy_model(validation_labels, soft_max_prediction)

# stats.mannwhitneyu
walk, walkhard, TP_real, TN_real, FP_real, FN_real = confusion_calc(validation_labels,soft_max_prediction)

#%%
#eyfp fos trap reactivation experiment
#recall counts
true_recall_mean, true_recall_std, true_recall = recall_calc(TP_real, FN_real)
lazy_recall_mean, lazy_recall_std, lazy_recall = recall_calc(TP_lazy, FN_lazy)
shuff_recall_mean, shuff_recall_std, shuff_recall = recall_calc(TP_shuff, FN_shuff)
#precision counts
true_prec_mean, true_prec_std, true_prec = precision_calc(TP_real, FP_real)
lazy_prec_mean, lazy_prec_std, lazy_prec = precision_calc(TP_lazy, FP_lazy)
shuff_prec_mean, shuff_prec_std, shuff_prec = precision_calc(TP_shuff, FP_shuff)

#fscores
a_lazy = fl_calc(lazy_recall_mean,lazy_prec_mean)
b_shuff = fl_calc(shuff_recall_mean,shuff_prec_mean)
c_true = fl_calc(true_recall_mean, true_prec_mean)
#%%
#mean over all positions for each animal
def plot_predictions(ground, lazy, shuff, performance_metric):
    x_axis_font = 20
    y_axis_font = 20

    ground_truth = ground
    lazy_data = lazy
    shuff_data = shuff
    metric = performance_metric
    
    d,lazy_p = stats.mannwhitneyu(ground_truth, lazy_data)
    f,shuff_p = stats.mannwhitneyu(ground_truth, shuff_data)
    print(lazy_p)
    print(shuff_p)
    ground_mean = np.mean(ground_truth) 
    lazy_mean = np.mean(lazy_data)
    shuff_mean = np.mean(shuff_data)
    ground_std = np.std(ground_truth)
    lazy_std = np.std(lazy_data)
    shuff_std = np.std(shuff_data)
    
    # Create lists for the plot
    materials = ['Network', 'ZeroR', 'Shuffled']
    x_pos = np.arange(len(materials))
    CTEs = [ground_mean, lazy_mean, shuff_mean]
    error = [ground_std, lazy_std, shuff_std]
    w = 0.4   # bar width
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error,
           align='center',
           ecolor='black',
           color=['mediumaquamarine','cadetblue','lightgrey'],
           capsize=10,
           alpha = 1,
           width=w)
    

    top = .85
    y_range =  .85
    # What level is this bar among the bars above the plot?
    level = 1
    # Plot the bar
    bar_height = (y_range * 0.07 * level) + top
    bar_tips = bar_height - (y_range * 0.02)

    # Significance level
    plt.plot(
        [1, 1, 0, 0],
        [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')

    #shuff
    top_shuff = .95
    y_range_shuff =  .95
    # What level is this bar among the bars above the plot?
    level_shuff = 1
    # Plot the bar
    bar_height_shuff = (y_range_shuff * 0.07 * level_shuff) + top_shuff
    bar_tips_shuff = bar_height_shuff - (y_range_shuff * 0.02)
    
    plt.plot(
        [2, 2, 0, 0],
        [bar_tips_shuff, bar_height_shuff, bar_height_shuff, bar_tips_shuff], lw=1, c='k')

    if lazy_p < 0.001:
        sig_symbol = '***'
    elif lazy_p < 0.01:
        sig_symbol = '**'
    elif lazy_p < 0.05:
        sig_symbol = '*'
    elif lazy_p > 0.05:
        sig_symbol = 'N.S.'
    
    if shuff_p < 0.001:
        sig_symbol_shuff = '***'
    elif shuff_p < 0.01:
        sig_symbol_shuff = '**'
    elif shuff_p < 0.05:
        sig_symbol_shuff = '*'
    elif shuff_p > 0.05:
        sig_symbol_shuff = 'N.S.'
    
    #for lazy
    text_height = bar_height - (y_range * 0.01)
    plt.text((x_pos[0] +x_pos[1]) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k', fontsize=20)
    
    #for shuff
    text_height_shuff = bar_height_shuff - (y_range_shuff * 0.01)
    plt.text((x_pos[0] +x_pos[2]) * 0.75, text_height_shuff, sig_symbol_shuff, ha='center', va='bottom', c='k', fontsize=20)
    
    
    
    
    ax.set_ylabel(metric, fontsize = 20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(materials, fontsize = 20)
    ax.set_title('Classifier ' +  metric +' on Test Set', fontsize = 20)
    ax.yaxis.grid(False)
    print(sig_symbol)
    plt.ylim([0,1.1])
    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plot_' + str(metric) + '.pdf')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()





