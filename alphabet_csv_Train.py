

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore') 

sns.set()
# loading dataset..
def load_csv_dataset():
    dataset = pd.read_csv("ALphabets/A_Z Handwritten Data.csv").astype('float32')
    dataset.rename(columns={'0':'label'}, inplace=True)
    return dataset

def Sample_of_Dataset(dataset):
    # Splite data the X - Our data , and y - the perdict label
    X = dataset.drop('label',axis = 1)
    y = dataset['label']
    return X,y

def Split_and_Scale(X,y):
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    # scale data
    standard_scaler = MinMaxScaler()
    standard_scaler.fit(X_train)
    
    X_train = standard_scaler.transform(X_train)
    X_test = standard_scaler.transform(X_test)

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return X_train, X_test, y_train, y_test

def Training_model(y):
    clss = Sequential()
    clss.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    clss.add(MaxPooling2D(pool_size=(2, 2)))
    clss.add(Dropout(0.3))
    clss.add(Flatten())
    clss.add(Dense(128, activation='relu'))
    clss.add(Dense(len(y.unique()), activation='softmax'))

    clss.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return clss

def Fitting_model(clss,X_train,X_test,y_train,y_test):    
   
    history = clss.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=200, verbose=2)
    clss.save('Alphabet.model')
    scores = clss.evaluate(X_test,y_test, verbose=0)
    print("CNN Score:",scores[1])
    return scores, history

def Val_loss_and_Model_loss(history):
    plt.plot(history.history['loss'] )
    plt.plot(history.history['val_loss'] )
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

def val_accuracy_and_Model_accuracy(history):
    plt.plot(history.history['accuracy'] )
    plt.plot(history.history['val_accuracy'] )
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    
    dataset = load_csv_dataset()
    X,y = Sample_of_Dataset(dataset)
    X_train, X_test, y_train, y_test = Split_and_Scale(X,y)
    clss = Training_model(y)
    scores, history  = Fitting_model(clss,X_train,X_test,y_train,y_test)
    scores = clss.evaluate(X_test,y_test, verbose=0)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))
    Val_loss_and_Model_loss(history)
    val_accuracy_and_Model_accuracy(history)
    

