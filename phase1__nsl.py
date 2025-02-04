import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics  import  recall_score
from sklearn  import  metrics
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import seaborn as sns
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import random
import pickle
import operator
#TODO: find other metrics (Done), add more datasets, pruning during sending to Minor node.
from sklearn.ensemble import RandomForestClassifier
feature=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","attack","difficulty_level"]
          
print("feature", len(feature))
train_data=pd.read_csv('archive/nsl-kdd/KDDTrain+.txt',names=feature)
test_data=pd.read_csv('archive/nsl-kdd/KDDTest+.txt',names=feature)
train_data.drop(['difficulty_level'],axis=1,inplace=True)
test_data.drop(['difficulty_level'],axis=1,inplace=True)
train_attack = train_data.attack.map(lambda a: 0 if a == 'normal' else 1)
test_attack = test_data.attack.map(lambda a: 0 if a == 'normal' else 1)
print("Testttttttttt", test_data.shape, train_data.shape)
#print(test_data.iloc[1000])
train_data['attack_state'] = train_attack
test_data['attack_state'] = test_attack
test_data.drop(['attack'],axis=1,inplace=True)
train_data.drop(['attack'],axis=1,inplace=True)
std_scaler = StandardScaler()
def standardization(df,col):
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
    return df
numeric_col_train = train_data.select_dtypes(include='number').columns
numeric_col_test = test_data.select_dtypes(include='number').columns
trainlabel=train_data["attack_state"]
train_data.drop("attack_state", axis=1)
testlabel=test_data["attack_state"]
test_data.drop("attack_state", axis=1)
train_data = standardization(train_data,numeric_col_train)
test_data = standardization(test_data,numeric_col_test)
train_data["attack_state"]=trainlabel
test_data["attack_state"]=testlabel
train_data = train_data[train_data.service != "red_i"]
train_data = train_data[train_data.service != "urh_i"]
train_data = train_data[train_data.service != "http_8001"]
train_data = train_data[train_data.service !="aol"]
train_data = train_data[train_data.service != "http_2784"]
train_data = train_data[train_data.service != "harvest"]
train_data = pd.get_dummies(train_data,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")
test_data = pd.get_dummies(test_data,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")

y_train = train_data["attack_state"]
X_train = train_data.drop("attack_state", axis=1)
print('X_train shape:',X_train.shape,'\ny_train shape:',y_train.shape)
y_test = test_data["attack_state"]
X_test = test_data.drop("attack_state", axis=1)
print('X_test shape:',X_test.shape,'\ny_test has shape:',y_test.shape)
def printScore(expected, predicted):
    accuracy = accuracy_score(expected, predicted)
    recall = recall_score(expected, predicted, average='micro')
    precision = precision_score(expected, predicted , average='micro')
    f1 = f1_score(expected, predicted , average='micro')
    print("Accuracy:",accuracy)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F-Score:",f1)
    ##calculate_error_rate 
expected = y_test
model= SVC()#MLPClassifier()#
print("Training in progess ...")
#model.fit(X_train, y_train)
#print("Training is Finished ...")
#predicted = model.predict(X_test)
#printScore(expected,predicted)

from sklearn.manifold import TSNE
data_1000 = X_train
labels_1000 = y_train
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(data_1000)
 #creating a new data frame which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df_bin = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "attack_state"))
### TSNE of binary classification dataset.
# T-sne of original dataset
sns.FacetGrid(tsne_df_bin, hue="attack_state").map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.savefig("all_classes_KDD.pdf")
plt.show()

X_train = X_train.astype(np.float64)
y_train = y_train.astype(np.float64)
X_test = X_test.astype(np.float64)
#Ensemble Training where n = number of devices
def func1(y_pred):
    y_pred_max = []
    for i in range(len(y_pred[0])):
        count0 = 0
        count1 = 0
        for j in range(len(y_pred)):
            if y_pred[j][i] <= 0.5:
                count0 += 1
            else:
                count1 += 1
        if count0 == count1:
            m = random.randint(0,1)
        elif count0 > count1:
            m = 0
        else:
            m = 1
        y_pred_max += [m]
    return y_pred_max

y_pred_train = []
y_pred_test = []
    
n=5
best = [0., 0.]
avg = [0., 0.]
ensemble = [0., 0.]
each_model_acc = {}
def calculate_acc_model(model, Xtest_data, ytest_data):
    return model.score(Xtest_data.values, ytest_data.values)
def assign_model_for_each_device(m):
    #models=["MLP", "SVC", "RandomForestClassifier"]
    #m = random.choice(models)
    print("The selected model is ", m)
    if m == "MLP":
        model = MLPClassifier()
    elif m == "KNN":
        model =  KNeighborsClassifier(n_neighbors=3)
    elif m == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif m == "DecisionTreeClassifier":
        model = DecisionTreeClassifier()
    elif m == "GradientBoostingClassifier":
        model  = GradientBoostingClassifier()
    return model
        #devices[device][0] =model.fit(X_train.values,y_train.values)
devices={}#each item contains devices[device_id]=(model, own_Xtest_slice, own_ytest)
#Pick random device to be a Miner
for i in range(n):
    devices["model_"+str(i)]=None
miner_id = random.choice(list(devices.keys()))
best_acc_inRounds={}
#Number of Rounds
rounds = 5
#############################
round_window=5
print("Shapes of train and test data = ",X_train.shape, X_test.shape)
for round in range(rounds):
    print("Round ...", round)
    
    
    for i in range(n):
        models=["MLP", "KNN", "RandomForestClassifier", "DecisionTreeClassifier", "GradientBoostingClassifier"]
        model_name = random.choice(models)
        #Training and Testing slices for each device.
        start = int(i*len(X_train)/n); end = int((i+1)*len(y_train)/n)
        start_test=int(i*len(X_test)/n); end_test = int((i+1)*len(y_test)/n)
        #print("Test Length ",start_test, end_test, len(X_test))
        device = "model_"+str(i)
        devices["model_"+str(i)]=[assign_model_for_each_device(model_name), X_test[start_test:end_test], y_test[start_test:end_test], model_name]
        #print("devices->", devices)
        
        #if not os.path.isfile("model_"+str(models[i])+".pickle"):
        print("Model "+str(models[i])+", is start training ...")
        clf = devices["model_"+str(i)][0].fit(X_train[start:end].values,y_train[start:end].values)
        pickle.dump(devices[device][0], open("model_"+str(models[i])+".pickle","wb"))#+str(i)+".pickle","wb"))
        #else:
            #print("Load weights file ...")
            #clf= pickle.load(open("model_"+str(models[i])+".pickle","rb"))#+str(i)+".pickle","rb"))
        devices["model_"+str(i)]=[clf, X_test[start_test:end_test], y_test[start_test:end_test],model_name]
        train, test = clf.score(X_train[start:end].values,y_train[start:end].values), clf.score(X_test.values,y_test.values)
        test_sample= X_test.iloc[1000]
        predicted = clf.predict(X_test.values)
        print(printScore(predicted, y_test.values)) ############################
        print("this traffic flow is predicted as ", predicted[0])
        if(best[1] < test):
            best[0] = train; best[1] = test
        avg[0] += train; avg[1] += test

        y_pred_train += [clf.predict(X_train.values)]
        y_pred_test += [clf.predict(X_test.values)]
#print("devices ",devices)
    avg[0] = avg[0]/n; avg[1] = avg[1]/n
    devices_acc={}# accurcies of devices model that tested on miner's data
#Pick random device to be a Miner
    #print("Miner _ id", miner_id, devices[miner_id])
    print("MINER ID = ", miner_id)
    miner_model = devices[miner_id][0] #random.choice(list(devices.items()))
#miner_model = miner_model[0]
    miner_Xtest_data = devices[miner_id][1]
    miner_ytest_data = devices[miner_id][2]
    for device in devices:
        if(miner_id != device):
            device_accuracy=0.0
            device_model = devices[device][0]
            device_Xtest_data = devices[device][1]
            device_ytest_data = devices[device][2]
            device_accuracy= device_model.score(miner_Xtest_data.values,miner_ytest_data.values)
            m_name = devices[device][3]
            model =devices[device][0]
            print("MODEL NAME = ", m_name)
            if m_name not in each_model_acc:
                each_model_acc[m_name] = [calculate_acc_model(model, device_Xtest_data, device_ytest_data)]
            else:
                each_model_acc[m_name].append(calculate_acc_model(model, device_Xtest_data, device_ytest_data))
            devices_acc[device] = device_accuracy
            if device in best_acc_inRounds:
                best_acc_inRounds[device].append(device_accuracy)
            else:
                best_acc_inRounds[device]=[device_accuracy]
            print("device_accuracy ", device, device_accuracy)
    '''device_max_acc = max(devices_acc,  key=devices_acc.get)
    best_model = devices[device_max_acc][0]
    print("Max id , acc = ",device_max_acc, devices_acc[device_max_acc])
    #After getting higher accuracy of model, send this model to all devices in the network.
    for device in devices:
        devices[device][0] = best_model
        device_Xtest_data = devices[device][1]
        device_ytest_data = devices[device][2]
        device_accuracy= devices[device][0].score(miner_Xtest_data.values,miner_ytest_data.values)
        #devices_acc[device] = device_accuracy
        print("device_accuracy after assiging best model", device, device_accuracy)'''
    #Here we find average of accurecy for each device for eacg round_window which is 5 rounds
    if round ==round_window:
        print("CHECKEDDDDDDDD", round)
        #TODO
        #Assign new miner to device with MAX acc.
        #Tell other devices of the new miner
        #find average and best acc as results for the paper.
        for device in devices:
            if device != miner_id:
                mean_acc=0.0
                mean_acc = sum(best_acc_inRounds[device])/len(best_acc_inRounds[device])
                best_acc_inRounds[device]=mean_acc
        round_window +=5
        print("best_acc_inRounds ",best_acc_inRounds)
        print("Previous Minder is ", miner_id)
        miner_id = max(best_acc_inRounds,  key=best_acc_inRounds.get)
        print("New Minder is ", miner_id)
        best_acc_inRounds={}
        #After getting higher accuracy of model, send this model to all devices in the network.
        device_max_acc = max(devices_acc,  key=devices_acc.get)
        best_model = devices[device_max_acc][0]
        print("Max id , acc = ",device_max_acc, devices_acc[device_max_acc])
        
        for device in devices:
            devices[device][0] = best_model
            device_Xtest_data = devices[device][1]
            device_ytest_data = devices[device][2]
            device_accuracy= devices[device][0].score(miner_Xtest_data.values,miner_ytest_data.values)
        #devices_acc[device] = device_accuracy
            print("device_accuracy after assiging best model", device, device_accuracy)
        print("each_model_acc ", each_model_acc)
y_pred_max_train = func1(y_pred_train)
y_pred_max_test = func1(y_pred_test)
#bclf = BaggingClassifier(base_estimator=MLPClassifier(), n_estimators=n, bootstrap=True)
#bclf.fit(X_train,y_train)
m1 = 0
m2 = 0
sum1 = 0
sum2 = 0
'''for i in range(n):
    # print(bclf[i].score(X_train,y_train))
    if bclf[i].score(X_train,y_train) > m1:
        m1 = bclf[i].score(X_train,y_train)
    sum1 += bclf[i].score(X_train,y_train)

    if bclf[i].score(X_test,y_test) > m2:
        m2 = bclf[i].score(X_test,y_test)
    sum2 += bclf[i].score(X_test,y_test)
'''
for i in devices:
    # print(bclf[i].score(X_train,y_train))
    if devices[i][0].score(X_train.values,y_train.values) > m1:
        m1 = devices[i][0].score(X_train.values,y_train.values)
        print("Best device in training ", i)
    sum1 += devices[i][0].score(X_train.values,y_train.values)

    if devices[i][0].score(X_test.values,y_test.values) > m2:
        m2 = devices[i][0].score(X_test.values,y_test.values)
        print("Best device in testing ", i)
    sum2 += devices[i][0].score(X_test.values,y_test.values)
print('Best train:',m1)
print('Best test:',m2)
print('Avg train:',sum1/n)
print('Avg test:',sum2/n)
#print('Train:',bclf.score(X_train,y_train))
#print('Test:',bclf.score(X_test,y_test))

