from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv1D
from collections import deque
from keras.utils.np_utils import to_categorical

from sklearn.cross_validation import train_test_split
from scipy.io.wavfile import read
import numpy as np
import os, sys


def conv1d(seq_length):
   model = Sequential()
   model.add(Conv1D(filters = 2, kernel_size = 10, input_shape=(seq_length, 2), activation= 'relu'))
   model.add(Flatten())
   model.add(Dense(512, activation='relu'))
   model.add(Dense(2, activation='softmax'))
   return model

def load_wav(file_path):
   return np.array(read(file_path)[1], dtype=float)

def load_single_class(dir_path, label):
   X = []
   y = []
   for file_name in os.listdir(dir_path):
      vec = load_wav(dir_path + '/' + file_name)
      vec -= np.mean(vec)
      vec /= np.max(vec)
  
   X.append(vec)
   y.append(label)
 
   return X, y

def find_max_len(X):
   max_len = 0
   for vec in X:
      if vec.shape[0] > max_len:
         max_len = vec.shape[0]
   return max_len

def reshape_with_zero_padding(X, max_len):
   X_pad = []
   for vec in X:
      pad_len = max_len - vec.shape[0]
      pad_vec = np.zeros((pad_len, 2))
      X_pad.append(np.concatenate((vec, pad_vec), axis=0))
  
   return X_pad
  
def main():
   oClassADirectory = sys.argv[1]          # Command line argument 1 is the folder containing class A samples.
   oClassBDirectory = sys.argv[2]          # Command line argument 2 is the folder containing class B samples.
 
   X_a, y_a = load_single_class(oClassADirectory, 0)
   X_b, y_b = load_single_class(oClassBDirectory, 1)
 
   X = X_a + X_b
   seq_length = find_max_len(X)
   X = reshape_with_zero_padding(X, seq_length)
   X = np.array(X)
 
   y = to_categorical(np.array(y_a + y_b), 2)
 
   X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state = 4)
 
   model = conv1d(seq_length)
   model.compile(loss = 'categorical_crossentropy', optimizer='RMSprop', metrics=['mse', 'accuracy'])
 
   model.fit(
       X_tr,
       y_tr,
       validation_data = (X_va, y_va),
       batch_size = 2,
       nb_epoch = 20,
       shuffle = True,
       )
   
   model.save("Models/current.h5")
 
main()