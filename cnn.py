'''
This file is used for creating a model
and saving it for later use
Also this file is also used to plot 5 graph that indicates
the increse of model metrics over time
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from metrics import recall_score, precision_score, f1_score




def load_data_and_process() -> tuple:
    '''
    Download and process data used for training model
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    # rescale pixel intensity between 0 and 1
    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encoded target
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return (X_train, y_train), (X_test, y_test)


def create_model() -> Sequential:
    '''
    Add more hidden layers for model
    and create the model
    '''
    model = Sequential()
    # add a convolutional layer with kernel = (3, 3) and 32 filers
    model.add(Conv2D(32, (3, 3), 
                    input_shape=(28, 28, 1), 
                    activation='relu', 
                    kernel_initializer='he_uniform'))
    # add max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # add conv2d layer 
    model.add(Conv2D(64, (3, 3), activation='relu', 
                    kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', 
                    kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # add flatten layer
    model.add(Flatten())
    # add dense layer
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt,
                  metrics=['acc', recall_score, precision_score, f1_score])
    return model


class CustomCallback(Callback):
    '''
    This class used as check point
    when training a model
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        check at the training process of the model
        '''
        if logs.get('acc') > 0.97:
            print('\nReached %2.2f%% accuracy, so stopping training' % (0.97 * 100))
            self.model.stop_training = True


# create a model
model = create_model()

# load MINIST dataset from internet using keras API
(X_train, y_train), (X_test, y_test) = load_data_and_process()

# create an instance of Mectrics class as implemented above to be used
# as callback
cb = CustomCallback()

#train the model and get its history
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, callbacks=[cb])

# save the model
model.save('trained.h5')


# history of acc of train and test set
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['training', 'test'], loc='upper left')
plt.show()

# history of loss of train and test set
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['training', 'test'], loc='upper left')
plt.show()

# history of recall of train and test
plt.plot(history.history['recall_score'])
plt.plot(history.history['val_recall_score'])
plt.title('Recall')
plt.ylabel('recall')
plt.xlabel('epochs')
plt.legend(['training', 'test'], loc='upper left')
plt.show()

# history of precision of train and test set
plt.plot(history.history['precision_score'])
plt.plot(history.history['val_precision_score'])
plt.title('Precision')
plt.ylabel('precision')
plt.xlabel('epochs')
plt.legend(['training', 'test'], loc='upper left')
plt.show()

# history of f1 score of train and test set
plt.plot(history.history['f1_score'])
plt.plot(history.history['val_f1_score'])
plt.title('F1 Score')
plt.ylabel('f1 score')
plt.xlabel('epochs')
plt.legend(['training', 'test'], loc='upper left')
plt.show()

print('=======================================================')
print(history.history['acc'])
print(history.history['val_acc'])
print('=======================================================')
print(history.history['loss'])
print(history.history['val_loss'])
print('=======================================================')
print(history.history['recall_score'])
print(history.history['val_recall_score'])
print('=======================================================')
print(history.history['precision_score'])
print(history.history['val_precision_score'])
print('=======================================================')
print(history.history['f1_score'])
print(history.history['val_f1_score'])






