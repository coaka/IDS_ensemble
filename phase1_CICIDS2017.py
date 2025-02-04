from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
from sklearn.preprocessing import OneHotEncoder 
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, LSTM
from sklearn.preprocessing import StandardScaler
import pandas as pd # 
df = pd.DataFrame()
for dirname, _, filenames in os.walk('CICIDS2017'):
    for filename in filenames:
        if filename.endswith('.csv'):
            print('Reading dataset files...')#os.path.join(dirname, filename))
            df1 = pd.read_csv('CICIDS2017/'+filename)
            df = pd.concat([df, df1])
            del df1
            
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
nRowsRead = None 

nRow, nCol = df.shape
df = df.dropna()
#print(print(df. info()))
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

#imp_mean.fit(df)

#df = imp_mean.transform(df)
df_new = df[np.isfinite(df.iloc[:, :-1]).all(1)]
print(f'Tabel of rows {nRow} No. of {nCol} colom')
train, test=train_test_split(df_new,test_size=0.3, random_state=10)
#print(df_new.head())
#train.describe()
test.describe()
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

# Convert labels to one-hot encoding
onehotencoder = OneHotEncoder()
trainDep = train[' Label'].values.reshape(-1, 1)
trainDep = onehotencoder.fit_transform(trainDep).toarray()
testDep = test[' Label'].values.reshape(-1, 1)
testDep = onehotencoder.fit_transform(testDep).toarray()
# Prepare data for PCA
num_components = 10
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
train_y = trainDep[:, 0]
test_y = testDep[:, 0]
# Reshape data for CNN
num_selected_features = train_X_selected.shape[1]
train_X_selected = train_X_selected.reshape(train_X_selected.shape[0], num_selected_features, 1)
test_X_selected = test_X_selected.reshape(test_X_selected.shape[0], num_selected_features, 1)
# Create CNN model
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(num_selected_features, 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(32, activation='relu'))
cnn_model.add(Dense(16, activation='relu'))
cnn_model.add(Dense(8, activation='relu'))
cnn_model.add(Dense(4, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train CNN model
history_cnn = cnn_model.fit(train_X_selected, train_y, epochs=5, batch_size=32, validation_split=0.2)
# Evaluate CNN model
test_loss_cnn, test_accuracy_cnn = cnn_model.evaluate(test_X_selected, test_y)
print(f'CNN Test Accuracy: {test_accuracy_cnn}')

#   
# Packet Attack Distribution
#print(train[' Label'].value_counts())
#print(test[' Label'].value_counts())

'''
scaler = StandardScaler()

# extract numerical attributes and scale it to have zero mean and unit variance  
print("Scaling the data...")
cols = train.select_dtypes(include=['float64','int64']).columns
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))
sc_test = scaler.fit_transform(test.select_dtypes(include=['float64','int64']))

# turn the result back to a dataframe
sc_traindf = pd.DataFrame(sc_train, columns = cols)
sc_testdf = pd.DataFrame(sc_test, columns = cols)
# creating one hot encoder object 
onehotencoder = OneHotEncoder() 
print("Applying onehot encoding of the data...")
trainDep = train[' Label'].values.reshape(-1,1)
trainDep = onehotencoder.fit_transform(trainDep).toarray()
testDep = test[' Label'].values.reshape(-1,1)
testDep = onehotencoder.fit_transform(testDep).toarray()
train_X=sc_traindf
train_y=trainDep[:,0]

test_X=sc_testdf
test_y=testDep[:,0]
#Feature Selection
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier();

# fit random forest classifier on the training set
print("features selection model...")
rfc.fit(train_X, train_y);

# extract important features
score = np.round(rfc.feature_importances_,3)
importances = pd.DataFrame({'feature':train_X.columns,'importance':score})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

# plot importances
plt.rcParams['figure.figsize'] = (11, 4)
importances.plot.bar();
#Recursive feature elimination
print("Recursive feature elimination...")
from sklearn.feature_selection import RFE
import itertools

rfc = RandomForestClassifier()

# create the RFE model and select 20 attributes
rfe = RFE(rfc, n_features_to_select=20)
rfe = rfe.fit(train_X, train_y)

# summarize the selection of the attributes
feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), train_X.columns)]
selected_features = [v for i, v in feature_map if i==True]

print(selected_features)

a = [i[0] for i in feature_map]
train_X = train_X.iloc[:,a]
test_X = test_X.iloc[:,a]
'''

print("Done...")