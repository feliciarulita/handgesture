import numpy as np

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split

import tensorflow as tf

#load Dataset
dataset = loadtxt('/Users/feliciarulita/Documents/practice/project5_HandGesture/model/poseClassifier/keypointsCoordinate.csv', delimiter=',')
input = dataset[:,1:]
output = dataset[:,0]

#splitting train and test dataset
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size = 0.2, random_state=40)

#define keras model
#need to find the best model
model = Sequential([
    Dense(42,input_shape=(42,),activation='relu'),
    # Dropout(0.2),
    Dense(30, activation='relu'),
    # Dropout(0.2),
    Dense(10, activation='relu'),
    # Dropout(0.2),
    Dense(7, activation='softmax')
])

#compile the keras model
# model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the keras model on the dataset
model.fit(x=input, y=output, epochs=15, batch_size=100)

#evaluate the keras model
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy test: %.2f' % (accuracy*100))

#save model
model_save = '/Users/feliciarulita/Documents/practice/project5_HandGesture/model/poseClassifier/keypointsCoordinate.keras'
save = model.save(model_save)

