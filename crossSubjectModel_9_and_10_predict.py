import numpy as np 
import random
import scipy.io as scio
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,average,concatenate
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from numpy import linalg as la
from libtlda.tca import TransferComponentClassifier


def generate_predict_data(s):
    list = [0,1,2,3,4,5,6,7]
    filtered_test_data = np.load("./preprocessed_data/filted_test_data.npy")
    filtered_test_target = np.load("./preprocessed_data/filted_test_target.npy")    
    filtered_evaluate_data = np.load("./preprocessed_data/filted_evaluation_data.npy")
    test_data = data_mpc_alignment(np.array(filtered_test_data),np.array(filtered_evaluate_data),s)
    
    data,target = test_data_augmentation(np.array(filtered_test_data),np.array(filtered_test_target),1024,512)
    test_data = data[list]
    test_target = target[list]
    sample_num = 8*(np.shape(test_data)[1])
    training_sample_num = int(sample_num*0.7)
    x = np.reshape(test_data,[sample_num,12,1024,1])
    y = np.reshape(test_target,[sample_num,1])
    index = random.sample(range(0,training_sample_num),training_sample_num)
    x_train = x[index]
    y_train = y[index]-1  
    return (x_train,y_train)


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


def data_mpc_alignment(test_data,evaluate_data,s):
    test_subject_num = np.shape(test_data)[0]
    test_trail_num = np.shape(test_data)[1]
    evaluate_trail_num = np.shape(evaluate_data)[1]
    channel = np.shape(test_data)[2]   
    samples = np.shape(test_data)[3]
    
    R_source = np.zeros([channel,channel])
    R_target = np.zeros([channel,channel]) 
    
    for k in range(test_subject_num):
        test_data[k] = test_data[k]-np.mean(test_data[k],axis=0)+np.mean(evaluate_data[s],axis=0)
    
    for j in range(evaluate_trail_num):
        R_source = R_source + np.dot(evaluate_data[s,j],np.transpose(evaluate_data[s,j]))
    R_source = R_source/(float(evaluate_trail_num))
    
    for k in range(test_subject_num):
        for j in range(test_trail_num):
            R_target = R_target + np.dot(test_data[k,j],np.transpose(test_data[k,j]))
        R_target = R_target/(float(test_trail_num))  
        for j in range(test_trail_num):
            test_data[k,j] = np.dot(np.dot(R_source,la.inv(R_target)),test_data[k,j])    
        
    return test_data 



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


def merge_model(INPUT,i,subjectType="withinSubject"):    
    model_1 = CNN_submodel_1(INPUT,1.0)
    model_2 = CNN_submodel_2(INPUT,0.8)
    model_3 = CNN_submodel_3(INPUT,0.6)
    model_4 = CNN_submodel_4(INPUT,0.5)
    
    if subjectType=="withinSubject":
        model_1.load_weights("./withinSubjectModel_saved/within_model_1_"+str(i+1)+".h5")
        model_2.load_weights("./withinSubjectModel_saved/within_model_2_"+str(i+1)+".h5")
        model_3.load_weights("./withinSubjectModel_saved/within_model_3_"+str(i+1)+".h5")
        model_4.load_weights("./withinSubjectModel_saved/within_model_4_"+str(i+1)+".h5")
    else:
        model_1.load_weights("./crossSubjectModel_saved/cross_model_1_"+str(i+1)+".h5")
        model_2.load_weights("./crossSubjectModel_saved/cross_model_2_"+str(i+1)+".h5")
        model_3.load_weights("./crossSubjectModel_saved/cross_model_3_"+str(i+1)+".h5")
        model_4.load_weights("./crossSubjectModel_saved/cross_model_4_"+str(i+1)+".h5")        
        
  
    y1 = model_1.layers[-2].output
    y2 = model_2.layers[-2].output
    y3 = model_3.layers[-2].output
    y4 = model_4.layers[-2].output

    y = concatenate([y1,y2,y3,y4])
    model = keras.Model(inputs = INPUT, outputs = y)
    return model

def creat_ensemble_model(INPUT,i,subjectType="withinSubject"):
    origin_model = merge_model(INPUT,i,subjectType=subjectType)
    for j in range(len(origin_model.layers)):
        origin_model.get_layer(index=j).trainable = False
        
    inp = origin_model.input
    y = origin_model.output
    ##  64，0.08, 
    dense = Dense(64,kernel_regularizer=keras.regularizers.l2(0.1))(y)
    dense = BatchNormalization(axis = 1)(dense)
    dense = Activation('elu')(dense)   
    dense = Dense(2, kernel_constraint = max_norm(0.015))(dense)
    result = Activation('softmax')(dense)
    return Model(inputs=INPUT, outputs=result)


def merge_result(data):
    result = []
    temp_origin = np.squeeze(data)
    temp  = np.reshape(temp_origin,[-1,7])
    average_temp = np.mean(temp,axis=1)
    for i in range((np.shape(average_temp))[0]):
        if average_temp[i]<0.5:
            result.append(0)
        if average_temp[i]>=0.5:
            result.append(1)
    return np.array(result)

def result_to_label(result):
    label = []
    for i in range(len(result)):
        if result[i][0]>0.5:
            label.append(0)
        if result[i][0]<0.5:
            label.append(1)  
    return np.array(label)


def CNN_submodel_1(INPUTS,l2):
    conv1 = Conv2D(32, (1, 128), padding = 'same',input_shape = (12, 1024,1))(INPUTS)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = DepthwiseConv2D((12, 1),depth_multiplier = 3,depthwise_constraint = max_norm(1.0))(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Activation('elu')(conv1)
    conv1 = AveragePooling2D((1, 16))(conv1)
    conv1 = Dropout(0.5)(conv1)
    
    conv2 = SeparableConv2D(64, (1, 64),padding = 'same')(conv1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Activation('elu')(conv2)
    conv2 = AveragePooling2D((1, 32))(conv2)
    conv2 = Dropout(0.5)(conv2)
        
    flatten = Flatten()(conv2)
    dense = Dense(16,activation = "elu",kernel_regularizer=keras.regularizers.l2(l2))(flatten)  
    dense = Dense(2, kernel_constraint = max_norm(0.01))(dense)
    result = Activation('softmax')(dense)
    
    return Model(inputs=INPUTS, outputs=result)


def CNN_submodel_2(INPUTS,l2):
    conv1 = Conv2D(32, (1, 128), padding = 'same',input_shape = (12, 1024,1))(INPUTS)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = DepthwiseConv2D((12, 1),depth_multiplier = 3,depthwise_constraint = max_norm(1.))(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Activation('elu')(conv1)
    conv1 = AveragePooling2D((1, 8))(conv1)
    conv1 = Dropout(0.5)(conv1)
    
    conv2 = SeparableConv2D(64, (1, 64),padding = 'same')(conv1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Activation('elu')(conv2)
    conv2 = AveragePooling2D((1, 8))(conv2)
    conv2 = Dropout(0.5)(conv2)
    
    conv3 = SeparableConv2D(64, (1, 64),padding = 'same')(conv2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Activation('elu')(conv3)
    conv3 = AveragePooling2D((1, 8))(conv3)
    conv3 = Dropout(0.5)(conv3)    
        
    flatten = Flatten()(conv3)
    dense = Dense(16,activation = "elu",kernel_regularizer=keras.regularizers.l2(l2))(flatten)    
    dense = Dense(2, kernel_constraint = max_norm(0.01))(dense)
    result = Activation('softmax')(dense)
    
    return Model(inputs=INPUTS, outputs=result)

def CNN_submodel_3(INPUTS,l2):
    conv1 = Conv2D(32, (1, 128), padding = 'same',input_shape = (12, 1024,1))(INPUTS)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = DepthwiseConv2D((12, 1),depth_multiplier = 3,depthwise_constraint = max_norm(1.))(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Activation('elu')(conv1)
    conv1 = AveragePooling2D((1, 4))(conv1)
    conv1 = Dropout(0.5)(conv1)
    
    conv2 = SeparableConv2D(64, (1, 16),padding = 'same')(conv1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Activation('elu')(conv2)
    conv2 = AveragePooling2D((1, 4))(conv2)
    conv2 = Dropout(0.5)(conv2)
    
    conv3 = SeparableConv2D(64, (1, 16),padding = 'same')(conv2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Activation('elu')(conv3)
    conv3 = AveragePooling2D((1, 4))(conv3)
    conv3 = Dropout(0.5)(conv3)    
    
    conv4 = SeparableConv2D(64, (1, 16),padding = 'same')(conv3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Activation('elu')(conv4)
    conv4 = AveragePooling2D((1, 4))(conv4)
    conv4 = Dropout(0.5)(conv4)        
        
    flatten = Flatten()(conv4)
    dense = Dense(16,activation = "elu",kernel_regularizer=keras.regularizers.l2(l2))(flatten)    
    dense = Dense(2, kernel_constraint = max_norm(0.01))(dense)
    result = Activation('softmax')(dense)
    
    return Model(inputs=INPUTS, outputs=result)


def CNN_submodel_4(INPUTS,l2):
    conv1 = Conv2D(32, (1, 128), padding = 'same',input_shape = (12, 1024,1))(INPUTS)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = DepthwiseConv2D((12, 1),depth_multiplier = 3,depthwise_constraint = max_norm(1.))(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Activation('elu')(conv1)
    conv1 = AveragePooling2D((1, 4))(conv1)
    conv1 = Dropout(0.5)(conv1)
    
    conv2 = SeparableConv2D(64, (1, 16),padding = 'same')(conv1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Activation('elu')(conv2)
    conv2 = AveragePooling2D((1, 4))(conv2)
    conv2 = Dropout(0.5)(conv2)
    
    conv3 = SeparableConv2D(64, (1, 16),padding = 'same')(conv2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Activation('elu')(conv3)
    conv3 = AveragePooling2D((1, 4))(conv3)
    conv3 = Dropout(0.5)(conv3)    
    
    conv4 = SeparableConv2D(64, (1, 16),padding = 'same')(conv3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Activation('elu')(conv4)
    conv4 = AveragePooling2D((1, 4))(conv4)
    conv4 = Dropout(0.5)(conv4)              
    
    conv5 = SeparableConv2D(64, (1, 16),padding = 'same')(conv4)
    conv5 = BatchNormalization(axis = 1)(conv5)
    conv5 = Activation('elu')(conv5)
    conv5 = AveragePooling2D((1, 4))(conv5)
    conv5 = Dropout(0.5)(conv5)     

    flatten = Flatten()(conv5)
    
    dense = Dense(16,activation = "elu",kernel_regularizer=keras.regularizers.l2(l2))(flatten)    
    dense = Dense(2, kernel_constraint = max_norm(0.01))(dense)    
    result = Activation('softmax')(dense)   
    
    return Model(inputs=INPUTS, outputs=result)   






### 验证精度
INPUTS = keras.Input(shape=(12,1024,1))

l2_value = 0.068457
mu_value = 1.218384
num_components_value = 9
bandwidth_value = 1.608087
filtered_evaluate_data = np.load("./preprocessed_data/filted_evaluation_data.npy")

for s in range(8,10):
    x_train,y_train = generate_predict_data(s)   
    x_test = evaluate_data_augmentation(filtered_evaluate_data,1024,512)[s]
    ensemble_CNN_model = creat_ensemble_model(INPUTS,s,subjectType="crossSubject")
    ensemble_CNN_model.load_weights("./crossSubjectModel_saved/MCNN_cross_model_"+str(s+1)+".h5")     
    ensemble_CNN_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', 
                  metrics=['acc'])
    new_ensemble_model = Model(inputs=ensemble_CNN_model.input, outputs=ensemble_CNN_model.layers[-3].output)   


    new_x_train = new_ensemble_model.predict(x_train)
    new_x_test = new_ensemble_model.predict(x_test)

    index = random.sample(range(3000),560)
    new_train_data = new_x_train[index]
    new_train_label = np.squeeze(y_train)[index]
    new_test_data = new_x_test


    TCA_classifier = TransferComponentClassifier(loss='logistic',l2=0.9,mu=mu_value,num_components=num_components_value,
                                                    kernel_type='rbf',bandwidth=bandwidth_value,order=2.0,)
    TCA_classifier.fit(new_train_data,new_train_label,new_test_data)
    y_pred = TCA_classifier.predict(new_test_data)


    merge_predict_label = merge_result(y_pred)        

    print("the predict label of the"+str(s)+"-th subject are:")
    print(merge_predict_label+1)
 
    print("____________________________________END_____________________________________________")    