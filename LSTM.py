import pandas as pd
import numpy as np
from array import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

airData = pd.read_csv("processedTrain.csv", usecols=[1,2,3,4,5,6,7,8,9], sep="," )
# print(airData)
# print(airData.shape)

nrows, ncols = airData.shape
trainingData = []
trainingTarget = []

for i in range(nrows):
	oneRow = airData.iloc[i]
	
	trainingTarget.append( float(oneRow["target"]) )
	
	pattern2D = []
	pattern2D.append(list( map(float, oneRow['temp'].split(',') )))
	pattern2D.append(list( map(float, oneRow['precip'].split(',') )))
	pattern2D.append(list( map(float, oneRow['rel_humidity'].split(',') )))
	pattern2D.append(list( map(float, oneRow['wind_dir'].split(',') )))
	pattern2D.append(list( map(float, oneRow['wind_spd'].split(',') )))
	pattern2D.append(list( map(float, oneRow['atmos_press'].split(',') )))
	trainingData.append(pattern2D)

# print( np.array(trainingData[0]) )
# print( np.array(trainingData[0]).shape )
# print( np.array(trainingData) )
print( np.array(trainingData).shape )

trainingData = np.array(trainingData)
sh1 = trainingData.shape[1]
sh2 = trainingData.shape[2]
trainingTarget = np.array(trainingTarget)
# print(trainingData)
# print(trainingTarget)
# print(type(trainingTarget[0]))

#Normalization
normed_matrix = normalize(trainingData[0], axis=1, norm='max')
print(normed_matrix)

model = Sequential()

model.add(LSTM(64, input_shape=trainingData.shape[1:], return_sequences=True) )
model.add(Dropout(0.2))

model.add(LSTM(64) )
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu') )
model.add(Dropout(0.2))

model.add(Dense(1, activation='linear') )

opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-5)

model.compile(loss="mse", optimizer=opt, metrics=['accuracy'] )

# model.fit(trainingData, trainingTarget, epochs=10, batch_size=32, validation_split=0.1)











