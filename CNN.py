import pandas as pd
import numpy as np
from array import *
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D

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

model = Sequential()

model.add(Conv1D(128, 3, input_shape=trainingData.shape[1:] ))
model.add(Activation("tanh"))
model.add(MaxPooling1D( pool_size=2 ) )
# model.add(Dropout(0.2))

model.add(Conv1D(128, 1))
model.add(Activation("tanh"))
model.add(MaxPooling1D( pool_size=1 ) )
# model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation("linear"))

model.compile( 	loss = "mse",
				optimizer = "adam",
				metrics = [tf.keras.metrics.RootMeanSquaredError()] )

model.fit( trainingData, trainingTarget, epochs=25, batch_size=32, validation_split=0.2, shuffle=True )

print(model.summary())








