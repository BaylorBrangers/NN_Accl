#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:56:47 2022

@author: baylor
"""



import numpy as np
import csv
import matplotlib.pyplot as plt
import re
from bisect import bisect_left
from scipy import signal
import tensorflow as tf
import functions_accl_extract as af
import class_functions_for_accl_extract as nacl
import os



f_accl=['weardata180206S00A1583.csv','weardata180206S00A1585.csv','weardata180208S00A1573.csv','weardata180308S04A1573.csv',
        'weardata180412S08A1585.csv', 'weardata180511S00A1693.csv','weardata2180511S00A1695.csv','weardata180511S00A1720.csv',
        'weardata180516S00A1736.csv','weardata180523S00A1694.csv','weardata180601S03A1693.csv','weardata180601S03A1695.csv',
        'weardata180601S03A1720.csv','weardata180607S03A1736.csv']

##Define annotation array
f_anno=['180206S00A1583.csv','180206S00A1585.csv','180208S00A1573.csv', '180308S04A1573.csv', '180412S08A1585.csv', 
        '180511S00A1693.csv','180511S00A1695.csv','180511S00A1720.csv','180516S00A1736.csv','180523S00A1694.csv',
        '180601S03A1693.csv','180601S03A1695.csv','180601S03A1720.csv','180607S03A1736.csv']



#f_accl=['weardata180206S00A1583.csv','weardata180206S00A1585.csv','weardata180208S00A1573.csv','weardata180308S04A1573.csv',
#        'weardata180412S08A1585.csv', 'weardata180511S00A1693.csv','weardata2180511S00A1695.csv','weardata180511S00A1720.csv']
#
#
#f_anno=['180206S00A1583.csv','180206S00A1585.csv','180208S00A1573.csv', '180308S04A1573.csv', '180412S08A1585.csv', 
#        '180511S00A1693.csv','180511S00A1695.csv','180511S00A1720.csv']


outfile = 'Train_Test_Data'
#from operator import itemgetter

data_dict={}
for f,file_name in enumerate(f_accl):
   name=str(file_name[7:])
   key=str(file_name[7:])
   print(name)
   name=nacl.accelerometer_profile()
   name.read_accelerometer_data(file_name)
   name.read_event_data(f_anno[f])
   data_dict[key] = name
   
training_data=[]
training_labels=[]


test_set=[]
test_labels=[]




validation=list(data_dict.keys())[3:8]

#for key,value in data_dict.items():
#    
#    #loop for all files and generate master training set
#    if key==validation:
#        v_labels_test=af.generate_Labels(value.indices,value.event_data,int(len(value.x_data)),'MountIn','MountOut')
#        
#        #foo1 and foo are not used, just didn't want to change the function. This generates data for the test set
#        foo,foo1,test_set_uni,test_labels_uni = af.window_data(value.x_data_corrected,value.y_data_corrected,
#                                                    value.z_data_corrected,v_labels_test,300)
#        if np.shape(test_labels)==(0,):
#            test_set=test_set_uni
#            test_labels=test_labels_uni
#        else:
#            test_set=np.concatenate((test_set,test_set_uni),axis=1)                                                                 
#            test_labels=np.concatenate((test_labels,test_labels_uni),axis=1)
#    
#    
#    else: 
#        #generate labels for test set
#        v_labels_train =af.generate_Labels(value.indices,value.event_data,int(len(value.x_data)),'MountIn','MountOut')
#        
#        #extract data for the test set and validation set
#        all_data, y_labels,all_data_gt, y_lables_gt = af.window_data(value.x_data_corrected,value.y_data_corrected,
#                                                        value.z_data_corrected,v_labels_train,300)
#        if np.shape(training_data)==(0,):
#            training_data=all_data
#            training_labels=y_labels
#        else:
#            training_data=np.concatenate((training_data,all_data),axis=1)                                                                 
#            training_labels=np.concatenate((training_labels,y_labels),axis=1)
#        

for key,value in data_dict.items():
    
    #loop for all files and generate master training set
    if key==validation:
        v_labels_test=af.generate_Labels(value.indices,value.event_data,int(len(value.x_data)),'MountIn','MountOut')
        
        #foo1 and foo are not used, just didn't want to change the function. This generates data for the test set
        foo,foo1,test_set_uni,test_labels_uni = af.window_data(value.x_data_corrected,value.y_data_corrected,
                                                    value.z_data_corrected,v_labels_test,300)
        if np.shape(test_labels)==(0,):
            test_set=test_set_uni
            test_labels=test_labels_uni
        else:
            test_set=np.concatenate((test_set,test_set_uni),axis=1)                                                                 
            test_labels=np.concatenate((test_labels,test_labels_uni),axis=1)
    
    
for key,value in data_dict.items():
    
    if key!=validation:
        #generate labels for test set
        v_labels_train =af.generate_Labels(value.indices,value.event_data,int(len(value.x_data)),'MountIn','MountOut')
        
        #extract data for the test set and validation set
        all_data, y_labels,all_data_gt, y_lables_gt = af.window_data(value.x_data_corrected,value.y_data_corrected,
                                                        value.z_data_corrected,v_labels_train,300)
        if np.shape(training_data)==(0,):
            training_data=all_data
            training_labels=y_labels
        else:
            training_data=np.concatenate((training_data,all_data),axis=1)                                                                 
            training_labels=np.concatenate((training_labels,y_labels),axis=1)
        








np.savez(outfile,test_set=test_set,test_labels=test_labels,training_data=training_data,training_labels=training_labels)  