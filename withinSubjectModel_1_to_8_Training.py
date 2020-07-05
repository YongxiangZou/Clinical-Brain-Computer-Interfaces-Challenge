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



def test_data_augmentation(data,label,length,step):
    '''
    Data augmentation for test data set by the slide window method.
    '''
    
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

def generate_within_training_data(i):
    '''
    Each subject has 80 trials, Among those trails, 16 trails are selected as the test set, the remaining 64 trials 
    are regraded as the training set. For the sake of category balance, 8 trails are randomly chosen 
    from ”left hand” category, the other 8 trails are randomly chosen from ”right hand” category    '''
    
    filted_data = np.load("./preprocessed_data/filted_test_data.npy")
    filted_target = np.load("./preprocessed_data/filted_test_target.npy")
    ### selecting 15 trails for test model, 65 trails for training
    trials_index = np.load("./withinSubjectModel_saved/subject_"+str(i+1)+"_trials_index.npy")
    test_trials_index = trials_index[:16]
    train_trials_index = trials_index[:]

    filted_test_x = filted_data[:,test_trials_index]
    filted_test_y = filted_target[:,test_trials_index]
    filted_train_x = filted_data[:,train_trials_index]
    filted_train_y = filted_target[:,train_trials_index]
  
    ## augmentation_selected_filted_data
    x_test,y_test = test_data_augmentation(np.array(filted_test_x),np.array(filted_test_y),1024,512) 
    x_train,y_train = test_data_augmentation(np.array(filted_train_x),np.array(filted_train_y),1024,512) 
    
    shuffle_num = np.shape(x_train)[1]
    shuffle_index = random.sample(range(0,shuffle_num),shuffle_num)
    x_train = x_train[:,shuffle_index]
    y_train = y_train[:,shuffle_index]
    return (x_train[i],y_train[i]-1,x_test[i],y_test[i]-1)


def merge_model(INPUT,i,subjectType="withinSubject"):   
    '''
    the flatten layers of four submodels are concatenated
    '''
    model_1 = CNN_submodel_1(INPUT,1.0)
    model_2 = CNN_submodel_2(INPUT,0.95)
    model_3 = CNN_submodel_3(INPUT,0.9)
    model_4 = CNN_submodel_4(INPUT,0.85)
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
    '''
    new full 
    '''
    origin_model = merge_model(INPUT,i,subjectType=subjectType)
    for j in range(len(origin_model.layers)):
        origin_model.get_layer(index=j).trainable = False        
    inp = origin_model.input
    y = origin_model.output
    ##  64，0.08, 
    dense = Dense(64,kernel_regularizer=keras.regularizers.l2(0.08))(y)
    dense = BatchNormalization(axis = 1)(dense)
    dense = Activation('elu')(dense)   
    dense = Dense(2, kernel_constraint = max_norm(0.02))(dense)
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
    dense = Dense(2, kernel_constraint = max_norm(0.07))(dense)
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
    dense = Dense(2, kernel_constraint = max_norm(0.05))(dense)
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
    dense = Dense(2, kernel_constraint = max_norm(0.04))(dense)
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
    dense = Dense(2, kernel_constraint = max_norm(0.04))(dense)
    result = Activation('softmax')(dense)
    return Model(inputs=INPUTS, outputs=result)   
   


print("During this process, it will take much time, and the submodels and the merged model are saved in withinSubjectModel_saved filefold")

INPUTS = keras.Input(shape=(12,1024,1))
for i in range(8):
    x_train ,y_train,x_test,y_test = generate_within_training_data(i) 
    for j in range(4):
        if j==0:
            current_submodel = CNN_submodel_1(INPUTS,1.1)
        if j==1:
            current_submodel = CNN_submodel_2(INPUTS,1.1)        
        if j==2:
            current_submodel = CNN_submodel_3(INPUTS,1.1)
        if j==3:
            current_submodel = CNN_submodel_4(INPUTS,1.1)          

### first stage of model training, the subCNN model is trained and saved
        current_submodel.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])
        current_submodel.fit(x_train, y_train,
                  batch_size=256,
                  epochs=1500,
                  validation_split=0.2)
        test_scores = current_submodel.evaluate(x_test,y_test, verbose=2)
        print('Test loss:', test_scores[0])
        print('Test accuracy:', test_scores[1])
        current_submodel.save("./withinSubjectModel_saved/within_model_"+str(j+1)+"_"+str(i+1)+".h5")
        
### second stage of model trainning, the MCNN model is trained and saved
for i in range(8):
    x_train ,y_train,x_test,y_test = generate_within_training_data(i) 
    ensemble_CNN_model = creat_ensemble_model(INPUTS,i,subjectType="withinSubject")
    ensemble_CNN_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    for j in range(len(ensemble_CNN_model.layers)):
        ensemble_CNN_model.get_layer(index=j).trainable = True    

    ensemble_CNN_model.fit(x_train, y_train,
              batch_size=256,
              epochs=1200,
              validation_split=0.2)
    test_scores = ensemble_CNN_model.evaluate(x_test,y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])          

    ensemble_CNN_model.save("./withinSubjectModel_saved/MCNN_within_model_"+str(i+1)+".h5")
