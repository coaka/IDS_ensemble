# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:59:18 2024

@author: Asus
"""

import numpy as np
import pandas as pd
import os
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
#from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from tensorflow.keras.layers import Concatenate, Flatten, Dense
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report, roc_auc_score, average_precision_score, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPool1D, Dropout, Input,MaxPooling1D
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import keras
# from keras.utils import normalize
from keras.utils.np_utils import normalize
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
from sklearn import linear_model
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint,EarlyStopping
###########
from sklearn.metrics  import  recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn  import  metrics
path='/'
#path='ddos_sdn/'

def printScore(expected, predicted):
    #expected=expected.argmax(axis=0)
    #predicted=predicted.argmax(axis=0)
    accuracy = accuracy_score(expected, predicted)
    recall = recall_score(expected, predicted, average='micro')
    precision = precision_score(expected, predicted , average='micro')
    f1 = f1_score(expected, predicted , average='micro')
    fpr, tpr, thresholds = metrics.roc_curve(expected, predicted)
    auc = metrics.roc_auc_score(expected, predicted,  average='micro')
    print('Results of RESNET model in NSL dataset with Classifier')
    print("Accuracy -->",accuracy)
    print("Precision -->",precision)
    print("Recall -->",recall)
    print("F-Score -->",f1)
    print("AUC -->", auc)
    # Open the file in append mode to add results
    #file.write(f"Iteration {i+1}:\n")
    with open(file_name, "a") as file:
        file.write(f"Accuracy: {accuracy:.4f} \n")
        file.write(f"Precision: {precision:.4f} \n")
        file.write(f"Recall: {recall:.4f} \n")
        file.write(f"F1: {f1:.4f}\n ")
        file.write(f"AUC: {auc:.4f} \n \n")
# Function to create a residual block
def residual_block(x, filters, kernel_size=3, strides=1):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv1D(filters, kernel_size, padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    
    if strides != 1:
        shortcut = Conv1D(filters, kernel_size=1, strides=strides, padding='same')(shortcut)
    
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x
####################
# Build the ResNet-like model using 1D convolutions
def RESNET_():
    input_layer = Input(shape=(X_train.shape[1],1))
    x = Conv1D(64, kernel_size=3, strides=1, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

# Add residual blocks
    x = residual_block(x, filters=64, strides=1)
    x = residual_block(x, filters=64, strides=1)
    x = residual_block(x, filters=128, strides=2)
    x = residual_block(x, filters=128, strides=2)

    x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    last = Dense(1, activation='sigmoid')(x)

    feature_extraction_model = Model(inputs=input_layer, outputs=last)

    return feature_extraction_model


# Load the NSL-KDD dataset
data = pd.read_csv('ddos_sdn/dataset_sdn.csv')
print(data.isna().any().any())

data = data.dropna()


print(data.isna().any().any())
data = data.drop(['dt','src','dst','Protocol'], axis=1)
y = data.label
train, test=train_test_split(data, test_size=0.3)
###############################PCA@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
missing_values_train = train.select_dtypes(include=['float64', 'int64']).isnull().sum()
missing_values_test = test.select_dtypes(include=['float64', 'int64']).isnull().sum()
numeric_cols_train=[]
numeric_cols_test=[]
if missing_values_train.any() or missing_values_test.any():
    # Handle missing values (imputation or removal)
    # For example, you can use imputation:
    numeric_cols_train = train.select_dtypes(include=['float64', 'int64']).columns
    print('numeric_cols_train->',numeric_cols_train)
    train[numeric_cols_train] = train[numeric_cols_train].fillna(train[numeric_cols_train].mean())
    numeric_cols_test = test.select_dtypes(include=['float64', 'int64']).columns
    test[numeric_cols_test] = test[numeric_cols_test].fillna(test[numeric_cols_test].mean())
# Check for extreme values
numeric_cols_train = train.select_dtypes(include=['float64', 'int64']).columns
numeric_cols_test = test.select_dtypes(include=['float64', 'int64']).columns
if ((train[numeric_cols_train] > np.finfo(np.float64).max).any().any() or
    (test[numeric_cols_test] > np.finfo(np.float64).max).any().any()):
    # Handle extreme values
    # Clip numeric columns
    train[numeric_cols_train] = train[numeric_cols_train].clip(lower=train[numeric_cols_train].quantile(0.01),
                                                              upper=train[numeric_cols_train].quantile(0.99),
                                                              axis=1)
    test[numeric_cols_test] = test[numeric_cols_test].clip(lower=test[numeric_cols_test].quantile(0.01),
                                                           upper=test[numeric_cols_test].quantile(0.99),
                                                           axis=1)
        
# Extract numerical attributes for scaling
scaler = StandardScaler()
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64', 'int64']))
sc_test = scaler.transform(test.select_dtypes(include=['float64', 'int64']))
print("Data before PCA", train.shape)

#####################
#print(df.columns)
plt.style.use('ggplot')

# Convert labels to one-hot encoding
onehotencoder = OneHotEncoder()
trainDep = train['label'].values.reshape(-1, 1)
trainDep = onehotencoder.fit_transform(trainDep).toarray()
testDep = test['label'].values.reshape(-1, 1)
testDep = onehotencoder.fit_transform(testDep).toarray()
# Prepare data for PCA
num_components = 18
pca = PCA(n_components=num_components)
# Fit and transform the training data
train_X_pca = pca.fit_transform(sc_train)
# Transform the testing data
test_X_pca = pca.transform(sc_test)
# Select features using SelectKBest with the f_classif scoring function
feature_selector = SelectKBest(score_func=f_classif, k='all')
# Fit and transform the training data
train_X_selected = feature_selector.fit_transform(train_X_pca, trainDep[:, 0])
# Transform the testing data
test_X_selected = feature_selector.transform(test_X_pca)
# Define target variables
y_train = trainDep[:, 0]
y_test = testDep[:, 0]
# Reshape data for CNN
num_selected_features = train_X_selected.shape[1]
X_train = train_X_selected.reshape(train_X_selected.shape[0], num_selected_features, 1)
X_test = test_X_selected.reshape(test_X_selected.shape[0], num_selected_features, 1)

nsamples, nx, ny = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny))
nsamples, nx, ny = X_test.shape
X_test = X_test.reshape((nsamples,nx*ny))

######################################################################
train_data=X_train 
test_data=X_test  
print("Data after PCA", X_train.shape)
##################
##################### classifier

classifier = GradientBoostingClassifier(
    n_estimators=100,      # Number of trees in the ensemble
    learning_rate=0.1,     # Learning rate for shrinkage
    max_depth=3,           # Maximum depth of each tree
    subsample=0.8,         # Subsample for reducing overfitting
    random_state=42        # Set random state for reproducibility
)
classifier2 = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    bootstrap_features=False,
    oob_score=True,  # Out-of-bag estimate
    random_state=42
)

classifier3 = DecisionTreeClassifier(
    criterion='entropy',
    splitter='best',
    max_depth=3,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
    )
##############################
##################
#############################
import keras
# from keras.optimizers import Adagrad AdamW
# optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

######################
file_name="metrics_results_SDN_1.txt"
with open(file_name, "a") as file:
    file.write("Results of SDNL dataset with Resnet only \n")  # 

nodes=5
for i in range(nodes):
 start = int(i*len(X_train)/nodes); end = int((i+1)*len(y_train)/nodes)
 start_test=int(i*len(X_test)/nodes); end_test = int((i+1)*len(y_test)/nodes)
 print("Training device->", i)
 
 
 resnet_model= RESNET_()
 resnet_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#####################
 ckpt_model='best_model_device_NSL_oversampling_'+str(i)+'.h5'
 checkpoint= ModelCheckpoint(ckpt_model,
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max',
                             verbose=1
                             )
 es = EarlyStopping(monitor='val_accuracy', patience=15,verbose=1)
 history = resnet_model.fit(X_train[start:end],
 y_train[start:end], validation_data=(X_test[start_test:end_test],y_test[start_test:end_test]),epochs=50
 ,batch_size=256,callbacks=[checkpoint, es])#callbacks=[checkpoint, es]
 resnet_model = load_model('best_model_device_NSL_oversampling_'+str(i)+'.h5',custom_objects={'CustomMaskLayer': CustomMaskLayer})
 ###########
 ###############

 predicted=resnet_model.predict(X_test[start_test:end_test])
 print('Results of NSL dataset with Resnet only ')
 predicted3 = predicted.round().astype(int)
 with open(file_name, "a") as file:
     file.write(f"node {i+1}:\n")
 
 printScore(y_test[start_test:end_test],predicted3)
 ####################################
 feature_extractor = Model(inputs=resnet_model.inputs,
 outputs=resnet_model.layers[-2].output)
 ####################
 features_train = feature_extractor.predict(X_train[start:end])
 features_test = feature_extractor.predict(X_test[start_test:end_test])
 ################
 ############
 print('Trainging with GradientBoostingClassifier')
 with open(file_name, "a") as file:
     file.write("Results of NSL dataset with GradientBoostingClassifier only \n")  # 
 classifier.fit(features_train, y_train[start:end])

 y_pred_classifier = classifier.predict(features_test)
 predicted3 = y_pred_classifier.round().astype(int)
 printScore(y_test[start_test:end_test],predicted3)
 ############
 print('Trainging with BaggingClassifier')
 with open(file_name, "a") as file:
     file.write("Results of NSL dataset with BaggingClassifier only \n")  # 
 classifier2.fit(features_train, y_train[start:end])

 y_pred_classifier = classifier2.predict(features_test)
 predicted3 = y_pred_classifier.round().astype(int)
 printScore(y_test[start_test:end_test],predicted3)
 ############
 print('Trainging with DecisionTreeClassifier')
 with open(file_name, "a") as file:
     file.write("Results of NSL dataset with DecisionTreeClassifier only \n")  # 
 classifier3.fit(features_train, y_train[start:end])

 y_pred_classifier = classifier3.predict(features_test)
 predicted3 = y_pred_classifier.round().astype(int)
 printScore(y_test[start_test:end_test],predicted3)
 print('End Node:'+str(i)+'End for')
 with open(file_name, "a") as file:
     file.write("################################################ \n")  # 


################ end for
################