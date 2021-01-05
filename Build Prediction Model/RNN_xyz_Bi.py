import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import time
import os

def MinMaxScaler(data):
    numerator = data - np.min(data,0)
    denominator = np.max(data,0) - np.min(data,0)
    return numerator/(denominator + 1e-7)

def InverseMinMaxScaler(data):
    
    return (np.max(test_set2[:,3:6],0)-np.min(test_set2[:,3:6],0)+1e-7)*data + np.min(test_set2[:,3:6],0)    

# features to use
#FEATURE_COLUMNS = ["Tension1(N)", "Tension2(N)", "Tension3(N)" , "Tension4(N)", "Tension5(N)", "Tension6(N)", "Tension7(N)", "Tension8(N)", "X_Err(mm)", "Y_Err(mm)", "Z_Err(mm)"]
FEATURE_COLUMNS = ["X_D(mm)", "Y_D(mm)", "Z_D(mm)", "X_Err(mm)", "Y_Err(mm)", "Z_Err(mm)", "X_R(mm)", "Y_R(mm)", "Z_R(mm)"]

# Window size or the sequence length
seq_length = N_STEPS =15

# load data
xy = pd.read_csv("Data_Train.txt", delimiter = "	") 
#xy = pd.read_csv("0.00135_Tension_Low_data.txt", delimiter = "	") 

data_set = xy.loc[:,FEATURE_COLUMNS].values       #remove time column

# 70% trainning set, 30% test set
train_size = int(len(data_set)*0.7)
train_set = data_set[0:train_size]
test_set2 = data_set[train_size :]              #Index from [train_size - seq_length] to utilize past sequence

# Feature Scaling
# Will use Normalisation as the Scaling function.
# Default range for MinMaxScaler is 0 to 1, which is what we want. So no arguments in it.

train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set2)    


#Build dataset
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series)-seq_length):
        _x = time_series[i : i+seq_length, 3:6]
        _y = time_series[i+seq_length,3:6]     #y_col_index
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

# reshape X to fit the neural network
trainX = trainX.reshape((trainX.shape[0], trainX.shape[2], trainX.shape[1]))
testX = testX.reshape((testX.shape[0], testX.shape[2], testX.shape[1]))

# # ==== Model =====================

model = Sequential()

# Adding the input layer and the LSTM layer
# Units memory units, sigmoid activation function and (None time interval with 1 attribute as input)
# first LSTM layer
model.add(Bidirectional(LSTM(units = 32, return_sequences = False, activation = 'tanh'), merge_mode = 'concat', input_shape = (None, seq_length)))
#model.add(Dropout(0.4))

# second LSTM layer
#model.add(Bidirectional(LSTM(units = 3, activation = 'tanh')))
#model.add(Dropout(0.4))

# Adding the output layer
# 1 nueron in the output layer for 1 dimensional output
# output layer
model.add(Dense(units = 3))
# Compiling the RNN
# Compiling all the layers together.
# Loss helps in manipulation of weights in NN. 
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

# Fitting the RNN to the Training set
# Number of epochs increased for better convergence.
history = model.fit(trainX, trainY, batch_size = 8, epochs = 100)

# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

model.save("Model_Bi_.h5")
#model.save("0.00135_Other_Low_model_Bi_.h5")
 
# # some tensorflow callbacks
# #checkpointer = ModelCheckpoint(os.path.join("results", model_name), save_best_only=True, verbose=1)
# #tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

# predictions
def predict(model, data, classification=False):
    prediction = model.predict(data)
    prediction = InverseMinMaxScaler(prediction)
    return prediction


# predict the future price
Prediction = predict(model, testX)
Experiment = InverseMinMaxScaler(testY)

#plot prediction
ft = 0      # 0 = x, 1 = y, 2 = z

plt.figure()
plt.plot(Experiment[:,ft], color = 'red', label = 'Experiment')

plt.plot(Prediction[:,ft], color = 'blue', label = 'Prediction')

plt.legend(loc = 'best')
plt.xlabel("Data number")
plt.ylabel("Error[mm]")
plt.show()

ft = 1      # 0 = x, 1 = y, 2 = z

plt.figure()
plt.plot(Experiment[:,ft], color = 'red', label = 'Experiment')

plt.plot(Prediction[:,ft], color = 'blue', label = 'Prediction')

plt.legend(loc = 'best')
plt.xlabel("Data number")
plt.ylabel("Error[mm]")
plt.show()

ft = 2      # 0 = x, 1 = y, 2 = z

plt.figure()
plt.plot(Experiment[:,ft], color = 'red', label = 'Experiment')

plt.plot(Prediction[:,ft], color = 'blue', label = 'Prediction')

plt.legend(loc = 'best')
plt.xlabel("Data number")
plt.ylabel("Error[mm]")
plt.show()

import math
from sklearn.metrics import mean_squared_error

# 0 = x, 1 = y, 2 = z
print ("x error:")
print (math.sqrt(mean_squared_error(Experiment[:, 0], Prediction[:, 0])))

print ("y error:")
print (math.sqrt(mean_squared_error(Experiment[:, 1], Prediction[:, 1])))

print ("z error:")
print (math.sqrt(mean_squared_error(Experiment[:, 2], Prediction[:, 2])))

#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#https://github.com/victor-wj/LSTM-GRU-and-BiLSTM-predict-IBM-stock-price/blob/master/Untitled.ipynb
#https://monkcage.github.io/blog/ai/2019/01/28/time_series_forecast.html
#https://www.kaggle.com/dimitreoliveira/deep-learning-for-time-series-forecasting
#https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/