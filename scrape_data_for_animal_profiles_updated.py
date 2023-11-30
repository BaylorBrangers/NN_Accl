#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:39:25 2022

@author: baylor
"""

import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
from bisect import bisect_left
from scipy import signal
import tensorflow as tf
import functions_accl_extract as af
import class_functions_for_accl_extract as nacl



rootdir = '/home/baylor/Documents/Animal_Accl_Event_Data/'

ext = ('wear')

all_animal_data= []

for root, dirs, files in os.walk(rootdir):

    for filename in files:
        '''Get session, id and create new accl object'''
        animal_id = root[-8:-3]
        session_id =root[-2:] 
        name = animal_id+session_id
        name=nacl.accelerometer_profile(animal_id,session_id)
        
        '''file order is event data first. functions within accl class rely on one another
        so this must be done first'''
        if not filename.startswith(ext):

            event_name = (root+ '/' + filename)
            print(event_name)
        else:
            '''create accl data extract,then create events. Event/index data are time 
            corrected wrt to start time offset on respective systems'''
            
            name.read_accelerometer_data(root+ '/' + filename)
            name.read_event_data(event_name)
            all_animal_data.append(name)
            
            

#%%

#mean mount time/animal/session number
session = [[],[],[],[],[],[]]

#total time spent in mount
session_total = [[],[],[],[],[],[]]

#mount attempts
session_MA = [[],[],[],[],[],[]]

session2 = np.array([[],[],[],[],[],[]],dtype= "object")
for i in all_animal_data:

    indvidual_sessions_data_name_event = i.event_data
    indvidual_sessions_data_time_event = i.time_data   
    sess = i.session
    animal = i.name
    n=0
    Mount= False
    mt=0
    mount_attempt_count = 0
    mount_time = []
    
    '''Below loop calculates the total mount time for each mountin/mount out event'''
    for n in np.arange(np.size(indvidual_sessions_data_time_event)):
        #all mount out must have mount in, this makes sure mount in/out are matched
 
        if np.str(indvidual_sessions_data_name_event[n]) == 'MountIn':
            
            mount_start= indvidual_sessions_data_time_event[n]
            Mount = True

        elif np.str(indvidual_sessions_data_name_event[n]) == 'MountOut':
 
            if Mount:
        
                mount_stop = indvidual_sessions_data_time_event[n]
                mt = mount_stop-mount_start
                mount_time =np.append(mount_time,mt)
                Mount = False
                mt = 0
    
        elif np.str(indvidual_sessions_data_name_event[n]) == 'MountAttempt':
        
            mount_attempt_count = mount_attempt_count + 1
        
        n=n+1

    '''calculate mean mount times per/animal during the session. Below will generate an array
    for meaned data for each animal'''
    
    #find the session number and index it for insertion into all sessions array
    mean_mount_pos = np.int(sess[1])-1
    
    #calculate the mean mount time of individual mouse for the duration of session in question
    mean_mount_time = np.mean(mount_time)
    total_mount_time = np.sum(mount_time)
    #add it 
    
    session[mean_mount_pos].append(mean_mount_time)    
    
    #total time spent in mounts
    session_total[mean_mount_pos].append(total_mount_time)
    
    #session2[mean_mount_pos] = np.append(session2[mean_mount_pos],mean_mount_time)
    
    #total number of mount attempts/session
    session_MA[mean_mount_pos].append(mount_attempt_count)
    



def time_first_intro(animal_data, behavior):    
    """Function for count number of occurences of a specific behavior""" 
    ar_first_intro = []
    for i in all_animal_data:
        indvidual_sessions_data_name_event = i.event_data
        indvidual_sessions_data_time_event = i.time_data   
        sess = i.session
        animal = i.name
        n=0
        for n in np.arange(np.size(indvidual_sessions_data_time_event)):
            if np.str(indvidual_sessions_data_name_event[n]) == behavior:
        
                ar_first_intro.append(indvidual_sessions_data_name_event[n])        
    
    return ar_first_intro



#plot mean mount times per/animal and across animals
plt.figure()
p=1
all_ses_mean = []
for t in session:
    all_ses_mean.append(np.mean(t))
    plt.scatter(np.repeat(p,np.size(t)),t)
    p=p+1
plt.plot(np.arange(1,np.size(all_ses_mean)+1,1),all_ses_mean)
plt.scatter(np.arange(1,np.size(all_ses_mean)+1,1),all_ses_mean,c ='black' )

plt.ylim([0,30])
plt.figure()

i=0
for i in range(0,4):
    tree = [row[i] for row in session]
    plt.plot(np.arange(1,np.size(all_ses_mean)+1,1),tree, 'ko', linestyle = 'solid')
    i = i +1
plt.ylim([0,50])
    
# plt.plot(np.arange(1,np.size(all_ses_mean)+1,1),all_ses_mean)
# plt.scatter(np.arange(1,np.size(all_ses_mean)+1,1),all_ses_mean,c ='black' )


