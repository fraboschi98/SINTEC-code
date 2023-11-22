# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:49:35 2023

@author: Utente
"""

import csv
import operator
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io
from scipy.signal import butter, find_peaks
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

from scipy import signal
from matplotlib import colors as mcolors
#=============================================================================#
# BASELINE REMOVAL
def baseline_removal(ecg_signal, window_size):
    ecg_signal=ecg_signal.flatten()
    #Moving average filter
    smoothed_signal = np.convolve(ecg_signal, np.ones(window_size)/window_size, mode='same')   
    #Baseline removal
    baseline_removed_signal = ecg_signal - smoothed_signal
    
    return baseline_removed_signal



# PEAKS DETECTION FUNCTION
def peaks_detection(s_filt,ts,time,th):
    
    pks=find_peaks(s_filt,height=th)  
    ind_pks=pks[0]
    ts_pks=np.zeros(len(ts))
    vect_pks=np.zeros(len(ts))
    vect_pks[ind_pks]=s_filt[ind_pks]        
    ts_pks[ind_pks]=ts[ind_pks]      

    # Local maximus deletion
    int_t=round(0.5/(time[-1]/len(time)))    
    for i in range(len(vect_pks)):
        if i>len(vect_pks)-int_t: 
            break
        if vect_pks[i]>0: 
            for j in range(1,int_t): 
                if vect_pks[i+j]>0: 
                    vect_pks[i+j]=0
                    ts_pks[i+j]=0
                    
    return vect_pks,ts_pks

# FEATURE REDUCTION FUNCTION
def feat_reduction(feat,t_fitted):
    
    row=np.zeros(len(t_fitted))
    T=0
    for i in range(len(t_fitted)-1):
        if i>=T:
            for j in range(i+1,len(t_fitted)):
                # Time window of 10 seconds
                if t_fitted[j]-t_fitted[i]>=10:
                    ind1=np.arange(i,j)
                    
                    # Feature averaging
                    vect_feat=feat[ind1]
                    val_feat=np.mean(vect_feat)
                    row[ind1]=val_feat
                    T=j
                    break

    # If the last window is smaller than 10 s, the last values are averaged and fitted in a 10 s time window
    for i in range(len(row)):
        if row[i]==0:
            ind1=np.arange(i,len(row))
            
            # Feature averaging
            val_feat=np.mean(feat[ind1])
            row[ind1]=val_feat
            break
        
    return row

# REGRESSION PROCESS FUNCTION
def regression_process(model,matr_train,matr_test,i_train,i_test,sbp,dbp):
    
    # SBP
    modelfit_SBP = model.fit(matr_train,sbp[i_train]) # Model training
    sbp_pred=modelfit_SBP.predict(matr_test)             # Model testing
    mae_sbp=mean_absolute_error(sbp[i_test], sbp_pred)           # SBP MEA (mmHg)
    dev_sbp=np.std(sbp_pred)                                        # SBP dev std (mmHg)
    err_sbp=abs(sbp_pred-sbp[i_test])
    n=np.array(np.where(err_sbp>5))
    num_sbp=len(np.transpose(n))
    
    # DBP
    modelfit_DBP = model.fit(matr_train,dbp[i_train]) # Model training
    dbp_pred=modelfit_DBP.predict(matr_test)             # Model testing
    mae_dbp=mean_absolute_error(dbp[i_test], dbp_pred)           # DBP MEA (mmHg)
    dev_dbp=np.std(dbp_pred)                                        # SBP dev std (mmHg)
    err_dbp=abs(dbp_pred-dbp[i_test])
    n=np.array(np.where(err_dbp>5))
    num_dbp=len(np.transpose(n))
    
    return modelfit_SBP,modelfit_DBP,sbp_pred,dbp_pred,mae_sbp,mae_dbp,dev_sbp,dev_dbp

#==============
# SIGNAL LOADING
#==============

#ECG
ecg_file_name = input("Enter the ECG .mat file name: ")
ecg_mat = scipy.io.loadmat(ecg_file_name)
ecg=ecg_mat['signal']
tsecg=ecg_mat['ts']

#PPG
ppg_file_name = input("Enter the PPG .mat file name: ")
ppg_mat = scipy.io.loadmat(ppg_file_name)
ppg=ppg_mat['signal']
tsppg=ppg_mat['ts']

#Omron
csv_file_name = input("Enter the Omron .csv file name: ")

fs_ecg = float(input("Enter the ECG sampling frequency (Hz): "))
fs_ppg = float(input("Enter the PPG sampling frequency (Hz): "))

'''
# Creation of numpy arrays
ecg_head=np.zeros(len(ecg))
ts_ecg=np.zeros(len(ecg))
ppg_head=np.zeros(len(ppg))
ts_ppg=np.zeros(len(ppg))
for i in range(len(ecg)):
    ecg_head[i]=ecg[i]
    ts_ecg[i]=tsecg[i]
    
for i in range(len(ppg)):
    ppg_head[i]=ppg[i]
    ts_ppg[i]=tsppg[i] 
'''
'''ecg_head=np.squeeze(ecg_mat['signal'])
ts_ecg=np.squeeze(ecg_mat['ts'])

ppg_head=np.squeeze(ppg_mat['signal'])
ts_ppg=np.squeeze(ppg_mat['ts'])'''

ecg_head = np.array(ecg).flatten()
ts_ecg= np.array(tsecg).flatten()
ppg_head = np.array(ppg).flatten()
ts_ppg= np.array(tsppg).flatten()

#OMRON
with open(csv_file_name) as filecsv:
    reader = csv.reader(filecsv, delimiter=";")
    ts_omron = np.array(list(map(float, [(line[0]) for line in reader])))

with open(csv_file_name) as filecsv:
    reader = csv.reader(filecsv, delimiter=";")
    sbp = np.array(list(map(float, [(line[1]) for line in reader])))

with open(csv_file_name) as filecsv:
    reader = csv.reader(filecsv, delimiter=";")
    dbp = np.array(list(map(float, [(line[2]) for line in reader])))


# Omron's time values correspond to the time in which the device returns the pressure values
ts_omron=ts_omron-60*np.ones(len(ts_omron)) 


rec_time_mins_ppg = ((len(ppg_head)-1)/fs_ppg)/60 
rec_time_mins_ecg = ((len(ecg_head)-1)/fs_ecg)/60

#=================
# SIGNAL FILTERING
#=================
#ECG
ecg_filt=baseline_removal(ecg_head, round(fs_ecg/2))

#PPG 
fNy = fs_ppg/2    # Nyquist frequency (Hz)
ft =10            # Cut off frequency (Hz) (experimental)
ws=0.1            # Passaband ripple (dB) (experimental)
wp=15             # Stopband attenuation (dB) (experimental)
fa=8              # Attenuation frequenzy (Hz) (experimental)
order=4           # Filter order (experimental)
n,wn=scipy.signal.buttord(ft/fNy,fa/fNy,ws,wp)
b,a=scipy.signal.butter(order,wn)                 #  low-pass Butterworth filter
ppg_filt1=scipy.signal.filtfilt(b,a,ppg_head)
ppg_filt=baseline_removal(ppg_filt1, round(fs_ppg/2))



#=================
# SIGNAL PREPARING
#=================

# Signal alignment 
tstart=ts_omron[0] #timestamp of the start sample of protocol registration

positive_diff_ppg = ts_ppg - tstart
positive_diff_ecg = ts_ecg - tstart


idx_ppg_start = np.where(positive_diff_ppg >= 0)[0][np.argmin(positive_diff_ppg[positive_diff_ppg >= 0])] #the index of the PPG sample with the least positive difference with the tstart sample is determined
idx_ecg_start = np.where(positive_diff_ecg >= 0)[0][np.argmin(positive_diff_ecg[positive_diff_ecg >= 0])] #the index of the ECG sample with the least positive difference with the tstart sample is determined


# Signal cut 
protocol_duration= 20 #minutes
idx_ppg_end=round(protocol_duration*60*fs_ppg+idx_ppg_start)
ppg_filt=ppg_filt[idx_ppg_start:idx_ppg_end] 
ts_ppg=ts_ppg[idx_ppg_start:idx_ppg_end]

idx_ecg_end=round(protocol_duration*60*fs_ecg+idx_ecg_start)
ecg_filt=ecg_filt[idx_ecg_start:idx_ecg_end] 
ts_ecg=ts_ecg[idx_ecg_start:idx_ecg_end]


t_ppg = np.arange(0,len(ppg_filt))/fs_ppg
t_ecg = np.arange(0,len(ecg_filt))/fs_ecg



#================
# KEY MEASUREMENTS FOR BP ESTIMATION: PTT AND HR
#================
#Peaks detection

plt.figure()
plt.plot(t_ecg,ecg_filt, label='Filtered ECG')
plt.ylabel('Amplitude (mV)')  
plt.xlabel('Time (minutes)')  
plt.legend()
plt.title('Select the optimal threshold for R-peaks detection:')
plt.show()

# R-peaks
th_ecg = float(input("Enter the threshold value for R-peaks detection from ECG: "))
vect_R, ts_R=peaks_detection(ecg_filt, ts_ecg, t_ecg, th_ecg)
indexR=np.nonzero(vect_R)[0] 




plt.figure()
plt.plot(t_ppg, ppg_filt, label='Filtered PPG')
plt.ylabel('Amplitude (LSD)')
plt.xlabel('Samples')
plt.title('Enter the threshold value for S-peaks detection from PPG: "')
plt.legend()
plt.show()

#S-peaks
th_ppg = float(input("Enter the threshold value for S-peaks detection from PPG: "))
vect_S, ts_S=peaks_detection(ppg_filt, ts_ppg, t_ppg, th_ppg)
indexP=np.nonzero(vect_S)[0] 

'''
plt.figure()
plt.plot(ppg_filt, label='Filtered PPG Signal')
plt.scatter(indexP, [ppg_filt[i] for i in indexP], color='red', marker='o', label='Detected P Peaks')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude[LSD]')
plt.title('PPG Signal with Detected S Peaks')
plt.legend()

plt.figure()
plt.plot( ecg_filt, label='Filtered ECG Signal')
plt.scatter(indexR, [ecg_filt[i] for i in indexR], color='red', marker='o', label='Detected R Peaks')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('ECG Signal with Detected R Peaks')
plt.legend()
'''

ts_S_no=np.extract(ts_S != 0, ts_S)
ts_R_no=np.extract(ts_R != 0, ts_R)


# PTT and HR
hr = [] 
timetable=[]
indexes=[]
ptt=[]
for ppg_timestamp in ts_S_no:
    
    index = np.searchsorted(ts_R_no, ppg_timestamp, side='right') - 1
    
  
    if index >= 0:
        indexes.append(index)
        ecg_before = ts_R_no[index]

        ptt_=ppg_timestamp-ecg_before
        ptt.append(ptt_)

        timetable_=ts_R_no[index] #ho tolto più1
        timetable.append(timetable_)
 

for i in range(indexes[0], len(ts_R_no) - 1):
    time_diff = ts_R_no[i + 1] - ts_R_no[i]
    if time_diff != 0:
        hr_ = 60 / time_diff
        hr.append(hr_)
    else:
        hr.append(0)

# Cleaning process
ptt = [x for x in ptt if x != 0]
hr = [x for x in hr if x != 0]
timetable = [x for x in timetable if x != 0]

# Ensure both ptt and hr have the same length
min_len = min(len(ptt), len(hr))
ptt = ptt[:min_len]
hr = hr[:min_len]
timetable = timetable[:min_len]
   

# Creation of the interpolating time array
n = np.where(ts_ecg == timetable[0])[0][0]
m = np.where(ts_ecg == timetable[-1])[0][0]
ind = np.arange(n, m)
t_fit = ts_ecg[ind]

# Interpolation of the Omron's data
SBP_fit=np.interp(t_fit,ts_omron,sbp)
DBP_fit=np.interp(t_fit,ts_omron,dbp)


# Interpolation of PTT and HR values
HR_fit=np.interp(t_fit,timetable,hr)
PTT_fit=np.interp(t_fit,timetable,ptt)

#==================
# SELECTION OF TRAIN AND TEST POINTS
#==================
row1=feat_reduction(PTT_fit,t_fit)     # PTT
row2=feat_reduction(HR_fit,t_fit)      # HR
row3=feat_reduction(SBP_fit,t_fit)     # SBP
row4=feat_reduction(DBP_fit,t_fit)     # DBP

# Arrays resampling
row1=np.interp(timetable,t_fit,row1)
row2=np.interp(timetable,t_fit,row2)
row3=np.transpose(np.interp(timetable,t_fit,row3))
row4=np.transpose(np.interp(timetable,t_fit,row4))

# Preparing data
# Training set contains the 70% of the whole dataset, the test set the remaining 30%
sz_train=round(0.7*len(row1))
ind_train=np.arange(0,sz_train)
ind_test=np.arange(sz_train,len(row1))
trainData_PTT=row1[ind_train]                       
trainData_HR=row2[ind_train] 

# Arrays cleaning
mean_PTT=np.mean(trainData_PTT)
dev_PTT=np.std(trainData_PTT)
mean_HR=np.mean(trainData_HR)
dev_HR=np.std(trainData_HR)

for i in range(len(ind_train)):
    if trainData_PTT[i]>mean_PTT+dev_PTT or trainData_PTT[i]<mean_PTT-dev_PTT or trainData_HR[i]>mean_HR+dev_HR or trainData_HR[i]<mean_HR-dev_HR:
        ind_train[i]=0
        

ind_train=[x for x in ind_train if x!=0]


trainData_PTT=row1[ind_train]                       
trainData_HR=row2[ind_train] 
testData_PTT = row1[ind_test]
testData_HR = row2[ind_test]
X_train=np.transpose(np.array([trainData_PTT,trainData_HR]))
X_test=np.transpose(np.array([testData_PTT,testData_HR]))
perc=round(0.2*len(ind_test))

#===================
# LINEAR REGRESSION
#===================

regr = linear_model.LinearRegression()             # Parameters definition
MLR_modelfit_SBP,MLR_modelfit_DBP,MLR_SBP_pred,MLR_DBP_pred,MLR_mae_SBP,MLR_mae_DBP,MLR_dev_SBP,MLR_dev_DBP=regression_process(regr,X_train,X_test,ind_train,ind_test,row3,row4)
MLR_coeff_SBP=MLR_modelfit_SBP.coef_
MLR_coeff_DBP=MLR_modelfit_DBP.coef_


total_minutes = (len(ind_test)*20)/len(ptt)
num_samples = len(ind_test)
x=np.linspace(0, total_minutes* 60, len(ind_test))

plt.figure()
# SBP plot
sbp_line, = plt.plot(x, MLR_SBP_pred, 'r', label="Predicted SBP",linewidth=2.5)
real_sbp_line, = plt.plot(x, row3[ind_test], 'b', label="Real SBP",linewidth=2.5)

# DBP plot
dbp_line, = plt.plot(x, MLR_DBP_pred, 'deeppink', label="Predicted DBP",linewidth=2.5)
real_dbp_line, = plt.plot(x, row4[ind_test], 'green', label="Real DBP",linewidth=2.5)

plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (mmHg)')
plt.legend()
plt.title('Results of BP prediction with SINTEC signals')

plt.show()


print(f"MLR MAE SBP: {round(MLR_mae_SBP, 1):.1f}")
print(f"MLR DEV SBP: {round(MLR_dev_SBP, 1):.1f}")
print(f"MLR MAE DBP: {round(MLR_mae_DBP, 1):.1f}")
print(f"MLR DEV DBP: {round(MLR_dev_DBP, 1):.1f}")


