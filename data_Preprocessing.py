
import numpy as np 
import random
import scipy.signal as signal
from scipy.signal import butter, lfilter
import scipy.io as scio
import os

def bandpass_filter(data, lowcut, highcut, fs, order):
    b,a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b,a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = butter(order,[low, high], btype = "bandpass")
    return b, a

def data_filter(data):
    for i in range (np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            for k in range(np.shape(data)[2]):
                data[i][j][k] = bandpass_filter(data[i][j][k],7,36,512,3)
    return np.array(data)


def test_data_augmentation(data,label,length,step):
    dim0 = np.shape(data)[0]
    dim1 = np.shape(data)[1]  
    dim2 = np.shape(data)[2]
    dim3 = np.shape(data)[3]  
    
    multiple = int((dim3-length)/step+1)
    new_dim1 = int(dim1*multiple)
    
    new_data = np.zeros([dim0,new_dim1,dim2,length])
    new_label = np.zeros([dim0,new_dim1,1])
    
    for i in range(dim0):
        for j in range(dim1):
            for n in range(multiple):
                new_data[i,j*multiple+n,:,:] = data[i,j,:,n*step:length+n*step]
                new_label[i,j*multiple+n,:] = label[i,j,:]
    result_data = np.reshape(new_data,[dim0,new_dim1,dim2,length,1])
    result_label = np.reshape(new_label,[dim0,new_dim1,1])
    return (result_data,result_label.astype(int))    

def evaluate_data_augmentation(data,length,step):
    #### 按照滑窗的方式对数据进行扩增
    dim0 = np.shape(data)[0]
    dim1 = np.shape(data)[1]  
    dim2 = np.shape(data)[2]
    dim3 = np.shape(data)[3]  
    
    multiple = int((dim3-length)/step+1)
    new_dim1 = int(dim1*multiple)
    new_data = np.zeros([dim0,new_dim1,dim2,length])
    
    for i in range(dim0):
        for j in range(dim1):
            for n in range(multiple):
                new_data[i,j*multiple+n,:,:] = data[i,j,:,n*step:length+n*step]
    result_data = np.reshape(new_data,[dim0,new_dim1,dim2,length,1])
    return (result_data)

raw_test_data = []
raw_test_target = []
raw_evaluate_data = []

path = "./data/"
files = os.listdir(path)
for file in files:
    if file[-5]=="T":
        raw_test_data.append(scio.loadmat(path+file)["RawEEGData"])
        raw_test_target.append(scio.loadmat(path+file)["Labels"])
    else:
        raw_evaluate_data.append(scio.loadmat(path+file)["RawEEGData"])

##对数据进行滤波处理
filted_test_data = data_filter(raw_test_data)
filted_test_target = raw_test_target
filted_evaluation_data = data_filter(raw_evaluate_data)

np.save("./preprocessed_data/filted_test_data.npy",filted_test_data)
np.save("./preprocessed_data/filted_test_target.npy",filted_test_target)
np.save("./preprocessed_data/filted_evaluation_data.npy",filted_evaluation_data)

##将滤波后的数据集进行增广
filted_test_data = np.load("./preprocessed_data/filted_test_data.npy")
filted_test_target = np.load("./preprocessed_data/filted_test_target.npy")
filted_evaluation_data = np.load("./preprocessed_data/filted_evaluation_data.npy")

augmentation_test_data,augmentation_test_target = test_data_augmentation(filted_test_data,filted_test_target,1024,512)
augmentation_evaluation_data = evaluate_data_augmentation(filted_evaluation_data,1024,512)

np.save("./preprocessed_data/augmentation_test_data.npy",augmentation_test_data)
np.save("./preprocessed_data/augmentation_test_target.npy",augmentation_test_target)
np.save("./preprocessed_data/augmentation_evaluation_data.npy",augmentation_evaluation_data)
