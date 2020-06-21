import pandas as pd
import numpy as np
from scipy import stats
import math
from array import *

airData = pd.read_csv("Train.csv", sep=",")
# print(airData)
print(airData.shape)

# oneRow = airData.iloc[0]
# print(oneRow['temp'])
# colarr = oneRow['temp'].split(',')
# colarr = list( map(float, colarr) )
# print(colarr)
# print(type(colarr[0]))

# numNans = 0;
# for index, row in airData.iterrows():
	# if('nan' in row['temp'] ):
		# numNans = numNans+1
# print(numNans)

temp_arr = []
precip_arr = []
humid_arr = []
dir_arr = []
spd_arr = []
atm_arr = []

for index, row in airData.iterrows():
	temp_arr.append(list( map(float, row['temp'].split(',') )) )
	precip_arr.append(list( map(float, row['precip'].split(',') )) )
	humid_arr.append(list( map(float, row['rel_humidity'].split(',') )) )
	dir_arr.append(list( map(float, row['wind_dir'].split(',') )) )
	spd_arr.append(list( map(float, row['wind_spd'].split(',') )) )
	atm_arr.append(list( map(float, row['atmos_press'].split(',') )) )

temp_arr = np.asarray(temp_arr)
precip_arr = np.asarray(precip_arr)
humid_arr = np.asarray(humid_arr)
dir_arr = np.asarray(dir_arr)
spd_arr = np.asarray(spd_arr)
atm_arr = np.asarray(atm_arr)

def replaceAllNaNs(inputArray):
	num_rows, num_cols = inputArray.shape
	for row in inputArray:					# Iterate the 2D array
		median = 0
		count = 0
		for x in row:						# Iterate the row array
			if (math.isnan(x)==False):
				median = median + x
				count = count+1
		median = median / count				# Calculate the median
		# print(median)
		index = 0
		for x in row:
			if (math.isnan(x)):
				row[index] = median			# Replace NaNs with median value
			index = index+1
		
	return inputArray

def fixSingularNaNs(inputArray):
	num_rows, num_cols = inputArray.shape
	for row in inputArray:					# Iterate the 2D array
		num_cols = row.shape[0]
		index = 0
		for x in row:						# Iterate the row array
			if (math.isnan(x)):
				if(index == 0):				# If the first value is a NaN
					fillNaN = row[index+1]	# Use the value from the next hour
					row[index] = fillNaN
					
				elif(index == num_cols-1):	# If the last value is a NaN
					fillNaN = row[index-2]	# Use the value from the previous hour
					row[index] = fillNaN
					
				else:
					if( math.isnan(row[index-1])==False and math.isnan(row[index+1])==False ):
						fillNaN = ( row[index-1] + row[index+1] )/2
						row[index] = fillNaN
						# print(fillNaN)
			index = index+1
	
	return inputArray

def checkForNaNs(inputArray):
	for row in inputArray:
		for x in row:
			if (math.isnan(x)):				# Check if there are remaining NaNs in the data (used for testing)
				return True
	
	return False
	
def standardizeColumn(inputArray):
	nrows, ncols = inputArray.shape
	print(inputArray.shape)
	
	allValues = []
	for row in inputArray:
		for x in row:
			allValues.append(x)				#Create 1D array of all values 
	
	allValues = np.asarray(allValues)
	print(allValues.shape)
	mean = np.mean(allValues)
	# print(mean)
	stdDev = np.std(allValues)
	# print(stdDev)
	
	index = 0
	for v in allValues:
		allValues[index] = (v-mean)/stdDev
		index = index+1
		
	# print(np.mean(allValues))
	# print(np.std(allValues))
	# print(allValues.shape)
	
	inputArray = np.reshape(allValues, (-1, 121))
	print(inputArray.shape)
	
	return inputArray

# print( np.array([temp_arr[2]]) )
# answer = replaceAllNaNs( np.array([temp_arr[2]]) )
# print(answer)

#Temperature
temp_arr = fixSingularNaNs(temp_arr)
temp_arr = replaceAllNaNs(temp_arr)
temp_arr = standardizeColumn(temp_arr)
print(checkForNaNs(temp_arr) )

#Precipitation
precip_arr = fixSingularNaNs(precip_arr)
precip_arr = replaceAllNaNs(precip_arr)
precip_arr = standardizeColumn(precip_arr)
print(checkForNaNs(precip_arr) )

#Relative Humidity
humid_arr = fixSingularNaNs(humid_arr)
humid_arr = replaceAllNaNs(humid_arr)
humid_arr = standardizeColumn(humid_arr)
print(checkForNaNs(humid_arr) )

#Wind direction
dir_arr = fixSingularNaNs(dir_arr)
dir_arr = replaceAllNaNs(dir_arr)
dir_arr = standardizeColumn(dir_arr)
print(checkForNaNs(dir_arr) )

#Wind Speed
spd_arr = fixSingularNaNs(spd_arr)
spd_arr = replaceAllNaNs(spd_arr)
spd_arr = standardizeColumn(spd_arr)
print(checkForNaNs(spd_arr) )

#Atmospheric pressure
atm_arr = fixSingularNaNs(atm_arr)
atm_arr = replaceAllNaNs(atm_arr)
atm_arr = standardizeColumn(atm_arr)
print(checkForNaNs(atm_arr) )

# Re-create dataset without missing data

# airDF = pd.DataFrame([], columns = ["ID", "location", "temp", "precip", "rel_humidity", "wind_dir", "wind_spd", "atmos_press", "target"])
# print(airDF)
nrows, ncols = airData.shape

def arrayToString(float_arr):
	strarr = '';
	for val in float_arr:
		strarr += str(val)
		strarr += ', '
	return strarr[0:len(strarr)-2]

# print(temp_arr[0] ) 
# print(arrayToString(temp_arr[0]) )
# oneRow = airData.iloc[0]
# print(oneRow['ID'])

newDataSet = []

for i in range(nrows):
	row = airData.iloc[i];
	# print([ row['ID'], 
			# row['location'], 
			# arrayToString(temp_arr[i]),
			# arrayToString(precip_arr[i]),
			# arrayToString(humid_arr[i]),
			# arrayToString(dir_arr[i]),
			# arrayToString(spd_arr[i]),
			# arrayToString(atm_arr[i]),
			# row['target'] ])
			
	newDataSet.append([ row['ID'], 
						row['location'], 
						arrayToString(temp_arr[i]),
						arrayToString(precip_arr[i]),
						arrayToString(humid_arr[i]),
						arrayToString(dir_arr[i]),
						arrayToString(spd_arr[i]),
						arrayToString(atm_arr[i]),
						row['target'] ])
	

newDataSet = np.asarray(newDataSet)
# print(newDataSet)

pd.DataFrame(newDataSet).to_csv("processedTrain.csv")






