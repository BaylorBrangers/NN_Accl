#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:41:12 2022

@author: baylor
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
from bisect import bisect_left
from scipy import signal
import tensorflow as tf

import random
np.random.seed(23)



####################################
def convert_to_tensor(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg
####################################
def accl_mag(accl_x, accl_y, accl_z):
    ##Calculates the magnitude of the acceleration vector
    mag_accl=np.sqrt(np.square(accl_x)+np.square(accl_y)+np.square(accl_z)) #3 to normalize

    return mag_accl
####################################

def read_accelerometer_data(filename):
    """ generate acclerometer data vector. Only looking at the
    first three components of the sensor"""

    with open (filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)

        timestamps = []
        #data from accelerometer
        x_data = []
        y_data = []
        z_data = []
        # contains data from y axis of gyroscope

        for line in reader:
            if line[1]=='34':
                timestamps.append(np.float(line[2]))
                x_data.append(np.float(line[3]))
                y_data.append(np.float(line[4]))
                z_data.append(np.float(line[5]))
            
            #for test files
#            timestamps.append(line[2])
#            x_data.append(line[3])
#            y_data.append(line[4])
#            z_data.append(line[5])
#            
        return (timestamps, x_data , y_data , z_data)

################################
####################################
def accl_mag(accl_x, accl_y, accl_z):
    ##Calculates the magnitude of the acceleration vector
    mag_accl=np.sqrt(np.square(accl_x)+np.square(accl_y)+np.square(accl_z)) #3 to normalize

    return mag_accl

####################################
def findClosest(myList, myNumber):
    pos = bisect_left(myList, myNumber)

    if pos == 0:
        return 0
    if pos == len(myList):
        return -1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return pos
    else:
       return (pos - 1)
 ####################################
def read_event_data(filename2):
    """  return event and time vectors for acquired data  """
    f = open(filename2, 'r')
    time_data = []
    event_data = []
    for line in f:
        t = line.strip().split()
        t1 = re.split(r'[T,+]' , t[1])

        t2 = t1[1].split(':')

        time_sec = int(t2[0]) * 3600 + int(t2[1]) * 60 + float(t2[2]) - 0.033

        event = t[0]

        time_data.append(time_sec)
        event_data.append(event)
    f.close()
    return (time_data , event_data)

#####################################

def generate_Labels(behavior_indicies,event_data,behavior_length,behavior1, behavior2):
   """returns an array of values [0,1], corresponding to when an
    animal was performing a certain behavior, as described by
    behavior1. Behavior2 is the stop behavior (eg, mount in/mount outhttps://www.pythonpool.com/python-loop-through-files-in-directory/)
    Used for training networks with labelled data"""

   event_index=behavior_indicies #array with timepoints for annotations
   length=int(behavior_length) #size of Label array for tflow
   Labels=np.zeros(length)
   final_stop='Ejaculation'
   start=[]
   stop=[]
   idx_check=0

   if behavior1 and behavior2:
       for counter,value in enumerate(event_data):
           if value==behavior1:
               start.append(event_index[counter])
               idx_check=+1
               Mount=True
           if value==behavior2 or value==final_stop:
               if len(start)==0:
                   continue
               else:
                   #if a mount out has occured and it was proceeded by a mount in
                   if event_index[counter] > start[idx_check-1] and Mount:
                       stop.append(event_index[counter])
                       Mount=False


       
       
       
   for b in range(len(start)):
       Labels[start[b]:stop[b]]=1

   return Labels


##########################################

def generate_Labels_all_Classes(behavior_indicies,event_data,behavior_length):
   """returns an array of values [0,1], corresponding to when an
    animal was performing a certain behavior, as described by
    behavior1. Behavior2 is the stop behavior (eg, mount in/mount out)
    Used for training networks with labelled data"""

   event_index=behavior_indicies #array with timepoints for annotations
   length=int(behavior_length) #size of Label array for tflow
   Labels=np.zeros(length)
   final_stop='Ejaculation'
   start=[]
   stop=[]
   idx_check=0





   for counter,value in enumerate(event_data):
       
       if value=='MountIn':
           idx_check=+1
           Mount=True
           start_behavior=event_index[counter]
           start.append(start_behavior)
       
       elif value=='MountAttempt':
           idx_check=+1
           MountAttempt=True    
           start_behavior=event_index[counter]
           start.append(start_behavior)
       
       #sniffing female will have a label of 3 
       elif value=='SniffingFemale' or value=='Annogenital':
           start_behavior=event_index[counter]
           Labels[start_behavior-20:start_behavior+10]=3
          
    #if ejac or end of mount attempt/mount   
       if value=='MountOut' or value==final_stop:
           if len(start)==0:
               continue
           else:
            
                #successful mounts have a label of 1   
               if event_index[counter] > start[idx_check-1] and Mount:
                   stop_behavior=event_index[counter]
                   Labels[start_behavior:stop_behavior]=1
                   Mount=False
                
                #mount attempts have a label of 2
               elif event_index[counter] > start[idx_check-1] and MountAttempt:
                   stop_behavior=event_index[counter]
                   Labels[start_behavior:stop_behavior]=2
                   MountAttempt=False
                       
   return Labels







#######################################################
    
def shuffle_data(accl,labels):
    a=np.random.permutation(np.size(accl,1)-1)
    naccl=np.transpose(accl)
    nlabels=np.transpose(labels)

    naccl=naccl[a]
    nlabels=nlabels[a]

    return naccl, nlabels

##########################################

def window_data(x,y,z,labels,window_size):

    tend=len(labels)
    wl=int(window_size)
    number_div=int(wl*3)
    window_divs=int(int(tend)/wl)

    accl=np.zeros((number_div,tend))
    y_label=np.zeros((2,tend)) #only 2 possible values for output


    for i in range(tend):
        if np.size(np.concatenate((x[i:i+wl], y[i:i+wl], z[i:i+wl]))) < number_div:
            break

        accl[:,i]=np.concatenate((x[i:i+wl], y[i:i+wl], z[i:i+wl]))
        
        
        if sum(labels[i:i+wl])==wl:
            y_label[:,i]=[1,0]
        
        #sexual behavior if more than half of the window
        elif sum(labels[i:i+wl])>(wl/4) and sum(labels[i:i+wl])!=0:
            y_label[:,i]=[1,0]
            
        #no sexual behavior if less than 1/2 window is not sex
        elif sum(labels[i:i+wl])<(wl/4) and sum(labels[i:i+wl])!=0:
            y_label[:,i]=[0,1]

        #if none of the label contain sex, nothing is happening    
        elif sum(labels[i:i+wl])==0:
            y_label[:,i]=[0,1]

    tr_accl=np.transpose(accl)
    tr_ylabel= np.transpose(y_label)
    return accl, y_label, tr_accl, tr_ylabel


####################################################
    
def window_data_all_behaviors(x,y,z,labels,window_size):

    tend=len(labels)
    wl=int(window_size)
    number_div=int(wl*3)
    window_divs=int(int(tend)/wl)

    accl=np.zeros((number_div,tend))
    y_label=np.zeros((2,tend)) #only 2 possible values for output


    for i in range(tend):
        if np.size(np.concatenate((x[i:i+wl], y[i:i+wl], z[i:i+wl]))) < number_div:
            break

        #each entry is a concatenation of all 3 axes of acceleration
        accl[:,i]=np.concatenate((x[i:i+wl], y[i:i+wl], z[i:i+wl]))
        
        #used for determining behavior makeup of the window
        window_label_range=labels[i:i+wl]
        
        #pure sexual behavior
        if sum(window_label_range)==wl and 3 not in window_label_range:
            y_label[:,i]=[1,0,0]
        
        #sexual behavior if more than half of the window
        elif sum(window_label_range)>(wl/4) and sum(window_label_range)!=0 and 3 not in window_label_range:
            y_label[:,i]=[1,0,0]
            
        #no sexual behavior if less than 1/2 window is not sex
        elif sum(window_label_range)<(wl/4) and sum(window_label_range)!=0:
            y_label[:,i]=[0,0,1]

        #if none of the label contain sex, nothing is happening    
        elif sum(window_label_range)==0:
            y_label[:,i]=[0,0,1]

    tr_accl=np.transpose(accl)
    tr_ylabel= np.transpose(y_label)
    return accl, y_label, tr_accl, tr_ylabel





#####################################################3
def window_data_oneaxis(x,labels,window_size):

    tend=len(labels)
    wl=int(window_size)
    number_div=int(wl)
    window_divs=int(int(tend)/wl)

    accl=np.zeros((number_div,tend))
    y_label=np.zeros((3,tend)) #only 3 possible values for output


    for i in range(tend):
        if np.size(x[i:i+wl]) < number_div:
            break

        accl[:,i]=(x[i:i+wl])

        if sum(labels[i:i+wl])==wl:
            y_label[:,i]=[1,0,0]

        if sum(labels[i:i+wl])<wl and sum(labels[i:i+wl])!=0:
            y_label[:,i]=[0,1,0]

        if sum(labels[i:i+wl])==0:
            y_label[:,i]=[0,0,1]

    tr_accl=np.transpose(accl)
    tr_ylabel= np.transpose(y_label)
    return accl, y_label, tr_accl, tr_ylabel




#####################################################3
def downsampling_the_accl(feature_vector,label_vector,down_sample):
    """Downsample the feature and label vectors for reducing size
    of input layer to the network. Down_sample is the down sampling
    rate (eg Down_sample=5 means every 5th point will be taken)"""

    if len(feature_vector)!=len(label_vector):
        print('Check feature and label, size mismatch detected')
        return
    new_feature=zeros((len(feature_vector,1)/down_sample),len(feature_vector,2))
    new_label=zeros((len(feature_vector,1)/down_sample),len(feature_vector,2))


#####################################################3
def windowing(feature_vector,label_vector,window_length):
    """Creates a sliding window for the network"""

    if len(feature_vector)!=len(label_vector):
        print('Check feature and label, size mismatch detected')
        return
    t0=0
    tend=len(label_vector)
    for i in range(tend):
        tstop=tstart+window_length
        if (tstop)>tend:
            tstop=tend
        tstart=+i

    return range
##############################################

def generate_test(accl,labels,percentage_for_test):
    """Split shuffled data into parts, based on percentage
    for test. This is used for cross-validation. Hardcoded
    for 80/20 currently
    """

    max_train_80=int(np.size(accl,0)*0.8)
    max_train_20=len(accl)-max_train_80

    #these value are hardcoded, change later
    train_x=np.zeros((max_train_80,300))
    train_y=np.zeros((max_train_80,3))
    test_x=np.zeros((max_train_20,300))
    test_y=np.zeros((max_train_20,3))

    train_x=accl[0:max_train_80,:]
    train_y=labels[0:max_train_80,:]
    test_x=accl[max_train_80:,:]
    test_y=labels[max_train_80:,:]



    return train_x, train_y, test_x, test_y


##################################################
def batch_dat_data(accl,labels,iteration,batch_size):
    """returns chunked data for mini-batch SGD.
    Make sure to shuffle data first"""

    b=batch_size
    i=iteration+1

    batch_x=accl[((i-1)*b):(i*b),:]
    batch_y=labels[((i-1)*b):(i*b),:]

    return batch_x,batch_y


##############################################

def model_out_to_pred(model_out,time_step):
    """converts output of model to binary prediction
    of mouse behavior. Used for model checking with
    accl data"""


    pred=model_out

    return pred

##############################################
def window_non_slide(x,y,z,labels,window_size):



    tend=len(labels)
    wl=int(window_size)
    number_div=int(wl*3)
    window_divs=int(int(tend)/wl)

    accl=np.zeros((number_div,window_divs))
    y_label=np.zeros((3,window_divs)) #only 3 possible values for output

    for i in range(window_divs):
        if np.size(np.concatenate((x[i:i+wl], y[i:i+wl], z[i:i+wl]))) < number_div:
            break

        accl[:,i]=np.concatenate((x[i*wl:(i+1)*wl], y[i*wl:(i+1)*wl], z[i*wl:(i+1)*wl]))

        if sum(labels[i*wl:(i+1)*wl])==wl:
            y_label[:,i]=[1,0,0]

        if sum(labels[i*wl:(i+1)*wl])<wl and sum(labels[i*wl:(i+1)*wl])!=0:
            y_label[:,i]=[0,1,0]

        if sum(labels[i*wl:(i+1)*wl])==0:
            y_label[:,i]=[0,0,1]

    accl=np.transpose(accl)
    y_label=np.transpose(y_label)

    #    y_label=np.concatenate((y_label,y_label,y_label),0)
    #    if sum(labels[i*wl:(i+1)*wl])==0
    return accl, y_label



##########################################################
def get_intromissions(behavior_indicies,event_data,behavior_length):
       """returns an array of values [0,1], corresponding to when an
        animal was performing intromissions
        Used for training networks with labelled data"""

       event_index=behavior_indicies #array with timepoints for annotations
       length=int(behavior_length) #size of Label array for tflow
       Labels=np.zeros(length)
       start=[]


       for counter,value in enumerate(event_data):
           if value=='Intromission':
               start.append(event_index[counter])

       for b in range(len(start)):
           Labels[start[b]-30:start[b]]=1

       return Labels



###############################################


def plot_subplot_sims(number_of_plots,data_1,data_2):

    colors=['xg','.-','bs']

    for i in range(number_of_plots):

        plt.subplot(2, 1, i)
        plt.plot(i,accuracy_test,'xg')
        plt.xlabel('Mini-batch')
        plt.ylabel('Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(i, avg_cost_array[epoch*total_batch+i], '.-')
        plt.xlabel('Mini-Batch')
        plt.ylabel('Cost')

    plt.show()

