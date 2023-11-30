#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 17:25:26 2022

@author: baylor
This code will plot the FFTs of the accelerometer data
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import re
from bisect import bisect_left
from scipy import signal
import tensorflow as tf
import keras as ks
import scipy as sp
import os
import functions_accl_extract as af

#%%

accl_data_path = '/home/baylor/Documents/Banas data/a1573/180208S00A1573/weardata.csv'

t,x,y,z = af.read_accelerometer_data(accl_data_path)
tree = np.asarray(x[1000:2024])
t= np.arange(1,1025)

a = [7899,7509,7231,7867,7730,7411,7479,7146,7367,8405]
b = [1,2,3,4,5,6,7,8,9,10]

# plt.plot(t,tree)

fs=100
amp = 2 * np.sqrt(2)
f, t, Zxx = sp.signal.stft(tree, fs, nperseg=1024)
plt.pcolormesh(t, f, np.abs(Zxx))
plt.title('STFT Magnitude')

plt.ylabel('Frequency [Hz]')

plt.xlabel('Time [sec]')

plt.show()