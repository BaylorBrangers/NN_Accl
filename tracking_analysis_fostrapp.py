#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 18:01:06 2022

@author: baylor
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import math
import os
import csv
import re
from bisect import bisect_left
from scipy import signal


filename = '/home/baylor/Documents/tracking for dreadds experiments/bb147/dcz/bb147_dcz.csv'
#import tracking data for both animals
#START AND STOP POINTS ACCORDING TO ANNOTATIONS
start_appetitive = 71500
start_sex = 105170
stop_sex =  177170

with open (filename, 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    next(reader, None)

    female = [[]]
    male = [[]]

    for line in reader:
        if line[0] == 'NaN':
            line[0] = 0.0
        if line[1] == 'NaN':
            line[1] = 0.0
        if line[2] == 'NaN':
            line[2] = 0.0
        if line[3] == 'NaN':
            line[3] = 0.0
        female.append([np.float(line[0]),np.float(line[1])])
        male.append([np.float(line[2]),np.float(line[3])])
    male.pop(0)
    female.pop(0)        


def cut_tracking(tracking_array, start_time,stop_time):
    '''returns a truncated array determined by the actual start and stop times of 
    tracking array. Start and stop times are located on google drive files'''
    
    new_tracking_array = tracking_array[start_time:stop_time]
    return new_tracking_array

def find_total_distance_traveled_single(pos_array):
    '''Find the single animal distance traveled'''
    individual_distances = []
    for position in range (0,len(pos_array)):
        if position != 0:
            individual_distances.append(math.dist(pos_array[position], 
                                                  pos_array[position-1]))
    return np.sum(individual_distances)


def find_total_distance_traveled_paired(pos_array1, pos_array2):
    '''Find the distance between paired position arrays. For between animals
    calculations. Position arrays must be the same length '''
    paired_distances = []
    for position in range (0,len(pos_array1)):
        if position != 0:
            paired_distances.append(math.dist(pos_array1[position], 
                                                  pos_array2[position]))
    return np.sum(paired_distances)



#calculate truncated arrays with correct start stop times
cut_male_all_phases = cut_tracking(male, start_appetitive, stop_sex)
cut_female_phases = cut_tracking(female, start_appetitive, stop_sex)
cut_female_appetitive = cut_tracking(female, start_appetitive , start_sex)
cut_male_appetitive = cut_tracking(male, start_appetitive, start_sex)
cut_feamle_consummatory = cut_tracking(female, start_sex, stop_sex)
cut_male_consummatory = cut_tracking(male, start_sex, stop_sex)


#Calculate total distances traveled for female and separation between them
#distance traveled individually appetive
total_distance_female_appetitve = find_total_distance_traveled_single(cut_female_appetitive)
total_distance_male_appetitve = find_total_distance_traveled_single(cut_male_appetitive)

#distance traveled individually consummatory
total_distance_female_consummatory = find_total_distance_traveled_single(cut_feamle_consummatory)
total_distance_male_consummatory = find_total_distance_traveled_single(cut_male_consummatory)

#distance during sex and non-sex
total_distance_btwn_MF_consummatory = find_total_distance_traveled_paired(cut_male_consummatory,
                                                                          cut_feamle_consummatory)
total_distance_btwn_MF_appetitive = find_total_distance_traveled_paired(cut_male_appetitive,
                                                                        cut_female_appetitive)
#%%
#plotting 

#total distance
sal_female_app = [143744.3106,110662.8991,64918.82279,6904.177727,181079.6756,131056.6745]
dcz_female_app = [104272.8264,32528.74689,66943.45042,63602.72909,253553.8012,312956.7733]

sal_female_cons = [303563.3942,143908.0119,41472.48157,602017.8031,304184.525,342023.9179]
dcz_female_cons = [259838.0483,201329.2511,274500.8688,588595.0197,449778.7488,611268.0293]

sal_male_app = [237930.9372,222278.7306,203326.8038,274.0682748,416936.8242,353209.4803]
dcz_male_app = [200011.8576,63465.19886,128421.7233,108675.6343,514634.2367,820047.8265]

sal_MF_app = [9405813,7474262.082,3727378.133,241687.7053,5540318.629,5066527.704]
dcz_MF_app = [9044612.96,2919066.013,3671027.876,1207606.253,6971484.85,10167804.41]

sal_MF_cons = [20922073.28,20293955.02,13432263.23,17083244.02,13080205.21,11229213.36]
dcz_MF_cons = [17137957.73,18822067.95,17703051.43,16539101.84,16584724.91,15187814.56]

#mean distance/frame
norm_sal_female_app = [4.257198596,3.857731965,4.125497127,1.071744447,7.25159888,4.993015638]
norm_dcz_female_app = [3.113179269,2.688548383,4.113016123,10.76370436,7.530555426,6.684682345]

norm_sal_female_cons = [4.216158253,1.998722388,0.5760066884,8.361358376,4.224785069,4.750332194]
norm_dcz_female_cons = [3.608861782,2.796239598,3.812512066,8.17493083,6.246927066,8.48983374]

norm_sal_male_app = [7.046673693,7.748683351,12.92112378,0.04254397311,16.6968413,13.45662452]
norm_dcz_male_app = [5.971572747,5.245491268,7.890250879,18.39154413,15.28465211,17.5160268]

norm_sal_MF_app = [278.566948,260.5543499,236.8694798,37.51749539,221.8701145,193.0252859]
norm_dcz_MF_app = [270.0368114,241.2650643,225.5485301,204.3672793,207.0533071,217.181887]

norm_sal_MF_cons = [290.5843511,281.8604863,186.5592115,237.2672781,181.6695168,155.9612967]
norm_dcz_MF_cons = [238.0271907,261.4176104,245.8757143,229.7097478,230.3434015,210.9418688]





#mount success rates
SAL = [4.257198596,3.857731965,4.125497127,1.071744447,7.25159888,4.993015638]

DCZ = [3.113179269,2.688548383,4.113016123,10.76370436,7.530555426,6.684682345]


SAL_mean = np.mean(norm_sal_MF_app)
SAL_std = np.std(norm_sal_MF_app)
DCZ_mean = np.mean(norm_dcz_MF_app)
DCZ_std = np.std(norm_dcz_MF_app)

t_stat_lateral,p_val=stats.ttest_ind(norm_sal_MF_app,norm_dcz_MF_app)


# Create lists for the plot
materials = ['Sal', 'DCZ',]
x_pos = np.arange(len(materials))
CTEs = [SAL_mean, DCZ_mean]
error = [SAL_std, DCZ_std]
w = 0.4   # bar width


#"darkgoldenrod" color used for consummatory, 'dodgerblue' used for appetitive
#'mintcream' used for controls


# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, 
        align='center', 
        ecolor='black', 
        color=['mintcream','dodgerblue'],
        capsize=10,
        alpha = 1,
        edgecolor='black',
        width=w)
a=x_pos[0] + np.random.random(np.size(DCZ)) * .2 - .1

b=x_pos[1] + np.random.random(np.size(DCZ)) * .2 - .1

i=0
t=0
s=.98
# 
for i in range(len(SAL)):
    ax.scatter(t, norm_sal_MF_app[i], color='black')
    ax.scatter(s, norm_dcz_MF_app[i], color='black')
    ax.plot((t,s),(norm_sal_MF_app[i],norm_dcz_MF_app[i]), color='black')


ax.set_ylabel('Pixels', fontsize = 16)
ax.set_xticks(x_pos)
ax.set_xticklabels(materials,fontsize = 16)
ax.set_title('Distance Between Male/Female -Appetitive Stage', fontsize = 16)

ax.yaxis.grid(False)
#code below generates significance bars for the plot. The Top variable control where the bars will be placed

bottom = 0
top = 300
y_range =  300
# What level is this bar among the bars above the plot?
level = 1
# Plot the bar
bar_height = (y_range * 0.07 * level) + top
bar_tips = bar_height - (y_range * 0.02)

# Significance level
plt.plot(
    [1, 1, 0, 0],
    [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
)
if p_val < 0.001:
    sig_symbol = '***'
elif p_val < 0.01:
    sig_symbol = '**'
elif p_val < 0.05:
    sig_symbol = '*'
elif p_val > 0.05:
    sig_symbol = 'N.S.'

text_height = bar_height + (y_range * 0.01)
plt.text((x_pos[0] +x_pos[1]) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k', fontsize=16)



plt.autoscale()
# Save the figure and show
plt.ylim([0,350])
plt.savefig('Distance_Female_Male_Appetitive.pdf')
plt.show()

print(p_val)
