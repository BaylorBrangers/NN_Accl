#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:56:51 2022

@author: baylor
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
from bisect import bisect_left
from scipy import signal
import tensorflow as tf
#import accl_functions as af
import random
np.random.seed(23)


class accelerometer_profile(object):
    """Creates a profile for each accelerometer profile"""
    def __init__(self, name,session):
        self.name = name
        self.session = session
        
    def read_accelerometer_data(self,file_name):
        """ generate acclerometer data vector. Only looking at the
        first three components of the sensor"""

        with open (file_name, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader, None)

            self.timestamps = []
            #data from accelerometer
            self.x_data = []
            self.y_data = []
            self.z_data = []
            # contains data from y axis of gyroscope

            for line in reader:
#                print(line)
                #check the positions in the file, sometimes indicies are wrong
                #current code is configd for banas data new, with a 3 in 
                #first column
                if line[1]=='34':
                    
                    self.timestamps.append(line[2])
                    self.x_data.append(line[3])
                    self.y_data.append(line[4])
                    self.z_data.append(line[5])
            
            #the first row contains column headers
            self.timestamps.pop(0)
            self.x_data.pop(0)
            self.y_data.pop(0)
            self.z_data.pop(0)

            #convert array data to float
            self.x_data = np.array([float(i) for i in self.x_data])
            self.y_data = np.array([float(i) for i in self.y_data])
            self.z_data = np.array([float(i) for i in self.z_data])

            self.timestamps = np.array([float(i.replace(',','.')) for i in self.timestamps])
            self.timestamps = np.array(self.timestamps)

            #still need to decide how to normalize data
            self.x_data_corrected=(self.x_data-np.mean(self.x_data))/np.var(self.x_data)
            self.y_data_corrected=(self.y_data-np.mean(self.y_data))/np.var(self.y_data)
            self.z_data_corrected=(self.z_data-np.mean(self.z_data))/np.var(self.z_data)
            
            
            return self.timestamps, self.x_data_corrected , self.y_data_corrected , self.z_data_corrected

    def findClosest(self,myList, myNumber):
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

    def read_event_data(self,filename2):
        """  return event and time vectors for acquired data  """
        f = open(filename2, 'r')
        self.time_data = []
        self.event_data = []
        self.indices = []
        self.acc_time_diff = np.zeros(len(self.timestamps))
        for line in f:
            t = line.strip().split()
            t1 = re.split(r'[T,+]' , t[1])

            t2 = t1[1].split(':')

            time_sec = int(t2[0]) * 3600 + int(t2[1]) * 60 + float(t2[2]) - 0.033

            event = t[0]

            self.time_data.append(time_sec)
            self.event_data.append(event)
        f.close()

        self.time_data = np.array(self.time_data)
        self.time_data_diff = np.zeros(len(self.time_data))
        
        #corrected time signal wrt to start of session
        for x in range (0, len(self.time_data)):
            self.time_data_diff[x] = self.time_data[x] - self.time_data[0]
        
         #correct for accl signal start time wrt to first time stamp   
        for x in range (0, len(self.timestamps)):
            self.acc_time_diff[x] = self.timestamps[x] - self.timestamps[0]
        
        #finds the position in the accl signal that matched behavior time annotation
        for x in self.time_data_diff:
            #self.indices.append(af.findClosest(self.acc_time_diff , x))
            self.indices.append(self.findClosest(self.acc_time_diff , x))
        
        #time data is time behavior occured, event is name of behavior, indcies is position on accl signal   
        return self.time_data , self.event_data, self.indices