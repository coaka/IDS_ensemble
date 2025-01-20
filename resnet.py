import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, Dense, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics  import  recall_score
from sklearn  import  metrics
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


def printScore(expected, predicted):
    expected=expected.argmax(axis=1)
    predicted=predicted.argmax(axis=1)
    accuracy = accuracy_score(expected, predicted)
    recall = recall_score(expected, predicted, average='micro')
    precision = precision_score(expected, predicted , average='micro')
    f1 = f1_score(expected, predicted , average='micro')
    fpr, tpr, thresholds = metrics.roc_curve(expected, predicted)
    auc = metrics.roc_auc_score(expected, predicted,  average='micro')
    
    print("Accuracy -->",accuracy)
    print("Precision -->",precision)
    print("Recall -->",recall)
    print("F-Score -->",f1)
    print("AUC -->", auc)
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

# Load the NSL-KDD dataset
data = pd.read_csv('ddos_sdn/dataset_sdn.csv')
print(data.isna().any().any())

data = data.dropna()


print(data.isna().any().any())
# Preprocess the data
for column in data.select_dtypes(include=['object']).columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])

# Split features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Convert target labels to categorical (one-hot encoding)
y = to_categorical(y)

# Reshape data for Conv1D: (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)

# Build the ResNet-like model using 1D convolutions
input_layer = Input(shape=(X_train.shape[1], 1))
x = Conv1D(64, kernel_size=7, strides=2, padding='same')(input_layer)
x = BatchNormalization()(x)
x = ReLU()(x)

# Add residual blocks
x = residual_block(x, filters=64, strides=2)
x = residual_block(x, filters=128, strides=2)
x = residual_block(x, filters=256, strides=2)
x = residual_block(x, filters=512, strides=2)

# Global Average Pooling and Dense output
x = GlobalAveragePooling1D()(x)
output_layer = Dense(y.shape[1], activation='softmax')(x)

# Build the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy:.4f}")
predicted = model.predict(X_test)
print(predicted)
predicted = predicted.astype(int)
printScore(y_test,predicted)
