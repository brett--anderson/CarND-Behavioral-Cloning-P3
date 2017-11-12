import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping

BATCH_SIZE = 256
DROP_RATE = 0.4
SIDE_FACTOR = 0.23
ACTIVATION = 'elu'
NUM_EPOCHS = 15
BASE_DATA_PATH = '../CarND-Behaviour-Data/my_data/'


# plot steering angles
# Try removing color learning

# Things to try next time...
# Try using transfer learning to add extra training data
# Try saving checkpoints after each epoch to perform training during available time

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

samples = []
with open(BASE_DATA_PATH + 'driving_log.csv') as csvfile:
	has_header = csv.Sniffer().has_header(csvfile.read(1024))
	csvfile.seek(0)	# rewind
	reader = csv.reader(csvfile)
	if has_header:
		next(csvfile)
	for line in reader:
		for i in range(3):
			source_path = line[i]
			filename = source_path.split('/')[-1]
			current_path = BASE_DATA_PATH + 'IMG/' + filename
			factor = 0.0
			if (i == 1):
				factor = SIDE_FACTOR
			elif (i == 2):
				factor = -SIDE_FACTOR
			angle = float(line[3]) + factor
			# original image and angle
			samples.append((current_path, angle, False))
			# flipped image and angle
			samples.append((current_path, angle*-1.0, True))

sklearn.utils.shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				current_path = BASE_DATA_PATH + 'IMG/' + filename
				image = cv2.imread(batch_sample[0])
				if batch_sample[2]:
					image = cv2.flip(image, 1)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				images.append(image)
				angles.append(batch_sample[1])

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

ch, row, col = 3, 160, 320

import keras.backend.tensorflow_backend as K
K.clear_session()

model = Sequential()
# normalize the images between 1 and -1, centered at 0
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch)))
# crop out parts of the image that are not useful to deciding the steering
# angle and which may confuse the model.
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Allow the model to choose the appropriate color space
model.add(Convolution2D(3, kernel_size=(1, 1), strides=(1, 1), activation='linear'))
# Implementation of the Nvidia network for a self driving car
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation=ACTIVATION))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation=ACTIVATION))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation=ACTIVATION))
model.add(Convolution2D(64, 3, 3, activation=ACTIVATION))
model.add(Convolution2D(64, 3, 3, activation=ACTIVATION))
model.add(Dropout(DROP_RATE))
model.add(Flatten())
model.add(Dense(100, activation=ACTIVATION))
model.add(Dropout(DROP_RATE))
model.add(Dense(50, activation=ACTIVATION))
model.add(Dropout(DROP_RATE))
model.add(Dense(10, activation=ACTIVATION))
model.add(Dense(1))
model.summary()

steps_per_epoch = len(train_samples) // BATCH_SIZE
validation_steps = len(validation_samples) // BATCH_SIZE

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
	steps_per_epoch=steps_per_epoch,
	validation_data=validation_generator, 
	validation_steps=validation_steps, 
	callbacks=[early_stopping],
	nb_epoch=NUM_EPOCHS)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
