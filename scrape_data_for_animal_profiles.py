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

session = [[],[],[],[],[],[]]
session2 = np.array([[],[],[],[],[],[]],dtype= "object")
for i in all_animal_data:

    indvidual_sessions_data_name_event = i.event_data
    indvidual_sessions_data_time_event = i.time_data   
    sess = i.session
    animal = i.name
    print(sess)
    print(animal)
    i=0
    Mount= False
    mt=0
    mount_attempt_count = 0
    mount_time = []
    
    '''Below loop calculates the total mount time for each mountin/mount out event'''
    for i in np.arange(np.size(indvidual_sessions_data_time_event)):
        #all mount out must have mount in, this makes sure mount in/out are matched
 
        if np.str(indvidual_sessions_data_name_event[i]) == 'MountIn':
            
            mount_start= indvidual_sessions_data_time_event[i]
            Mount = True

        elif np.str(indvidual_sessions_data_name_event[i]) == 'MountOut':
 
            if Mount:
        
                mount_stop = indvidual_sessions_data_time_event[i]
                mt = mount_stop-mount_start
                mount_time =np.append(mount_time,mt)
                Mount = False
                mt = 0
    
        elif np.str(indvidual_sessions_data_name_event[i]) == 'MountAttempt':
        
            mount_attempt_count = mount_attempt_count + 1
        
        i=i+1
    
    '''calculate mean mount times per/animal during the session. Below will generate an array
    for meaned data for each animal'''
    
    #find the session number and index it for insertion into all sessions array
    mean_mount_pos = np.int(sess[1])-1
    
    #calculate the mean mount time of individual mouse for the duration of session in question
    mean_mount_time = np.mean(mount_time)
    
    #add it 
    session[mean_mount_pos].append(mean_mount_time)    
    session2[mean_mount_pos] = np.append(session2[mean_mount_pos],mean_mount_time)

# mean_val_array = []
# for t in np.arange(1,7):
#     na = 'session_' + str(t)
#     mean_val_array.append(np.mean(globals()[na]))

plt_range = np.arange(1,7)
# p=1
# for t in session:

#     plt.scatter(np.repeat(p,np.size(t)),t)
#     p=p+1

for t in session:
    plt.plot(p,np.mean(t))
    p=p+1

# for t in np.arange(6):
#     plt.plot(plt_range[t],mean_val_array[t])
#     print(plt_range[t],mean_val_array[t])
# plt.show()    


# plt.plot(plt_range,mean_val_array)
# p=1


# session_number_plotting = np.size(session,0)
# for i in session:
    
#    plt.scatter(p,i)


# plt.ylim([0,30])
   
# mean_session_1 = np.mean(session_1)
# mean_session_2 = np.mean(session_2)
# mean_session_3 = np.mean(session_3)
# mean_session_4 = np.mean(session_4)
# mean_session_5 = np.mean(session_5)
# mean_session_6 = np.mean(session_6)
  
# all_data_dict={}
# for i in np.arange(np.size(indvidual_sessions_data_name_event)):
#     if indvidual_sessions_data_name_event[i] in all_data_dict:
#         all_data_dict[indvidual_sessions_data_name_event[i]].append(indvidual_sessions_data_time_event[i])
    
#     else:
#         all_data_dict[indvidual_sessions_data_name_event[i]]=[]
#         all_data_dict[indvidual_sessions_data_name_event[i]].append(indvidual_sessions_data_time_event[i])
#     i = i +1
    
# all_data_dict['mount_times']=mount_time

# free=np.mean(np.array(all_data_dict['mount_times']))

             
            
    