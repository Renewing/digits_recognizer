'''
	digits algorithm
	@author: renewing
'''

import pandas as pd
import numpy as np
import time

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
ids = (pd.read_csv('sample_submission.csv'))['ImageId']

data_train.sample(3)
start_time = time.time()

def get_labels(df):
	# try:
	labels = data_train.label
	labels = labels.values
	del df['label']
	return labels, df
	# except:
		# return df
def get_mat(df):
	dataArr = np.zeros((42000,784))
	for i in range(42000):
		digits = df[i:i+1]
		digits_new = digits.values
		dataArr[i] = digits_new
	return dataArr

def	transform_data(df):
	labels, df = get_labels(df)
	dataArr = get_mat(df)
	return labels, dataArr
	
trainingLabels, trainingData = transform_data(data_train)
# data_test = transform_data(data_test)
# test = data_train[41998:41999].values
	
from sklearn import neighbors

# create the model
knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')

# fit the model
# knn.fit(trainingData, trainingLabels)

# result = knn.predict(test)

# print("the last number is:",result)
# print("train success!")

from sklearn.model_selection import train_test_split

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(trainingData, trainingLabels,
													test_size = num_test)

from sklearn.metrics import accuracy_score	
												
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
print("the accuracy is:", accuracy_score(y_test, predictions))

# predictions = np.zeros((28000,1))
# for i in range(28000):
	# predictions[i] = int(knn.predict(data_test[i:i+1].values))

predictions = knn.predict(data_test)	
output = pd.DataFrame({'ImageId' : ids, 'Label' : predictions})
output.to_csv('digits_predictions.csv', index=False)
output.head()
end_time = time.time()
print("the total time is:", end_time - start_time)