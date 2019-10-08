#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.datasets import fashion_mnist #파이썬으로 작성된 오픈소스 신경망 라이브러리.텐서플로 팀은 코어 라이브러리에 케라스를 지원하기로 함.
import numpy as np
import tensorflow
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

(train_X,train_Y),(test_X,test_Y)=fashion_mnist.load_data()
print('Training data shape : ',train_X.shape, train_Y.shape) #train_X는 (60000, 28, 28) train_Y는 (60000,)
print('Testing data shape : ', test_X.shape, test_Y.shape) #test_X는 (10000,28,28), train_Y는 (10000,)
classes = np.unique(train_Y)#나올 수 있는 class의 경우
nClasses = len(classes)#nClasses는 class의 길이
print('Total number of outputs: ',nClasses)
print('Output classes : ',classes)

plt.figure(figsize=[5,5])
#훈련 셋의 첫 번째 이미지 보여주기
plt.subplot(121)
plt.imshow(train_X[0,:,:],cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

#테스트 셋의 첫 번째 이미지 보여주기
plt.subplot(122)
plt.imshow(test_X[0,:,:],cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))

#convert 28*28 images to 28*28*1
train_X=train_X.reshape(-1,28,28,1)
test_X=test_X.reshape(-1,28,28,1)
print('train_x reshape: ',train_X.shape)
print('train_y reshape: ',train_Y.shape)
#data는 int8 format 이므로 float32로 변경
train_X=train_X.astype('float32')
test_X=test_X.astype('float32')
train_X= train_X/255
test_X=test_X/255
#normalization.
#rescale
train_Y_one_hot=to_categorical(train_Y)
test_Y_one_hot=to_categorical(test_Y)
#one-hot encoding vector로의 변형
#categorical data를 vector로 변경한 것.

print("Original label:",train_Y[0])
print('After conversion to one-hot :',train_Y_one_hot[0])

train_X,valid_X,train_label,valid_label = train_test_split(train_X,train_Y_one_hot,test_size=0.2, random_state=13)
print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)
#train_X set is 48000, valid_X set is 12000

#The first layer will have 32-3 x 3 filters,
#The second layer will have 64-3 x 3 filters
#The third layer will have 128-3 x 3 filters.
#In addition, there are three max-pooling layers each of size 2 x 2.
#Flatten(fully_connected_layer)
#dense_layer(?) ->마지막 출력되는 것이기 때문에
#output layer

batch_size=64
epochs=20
num_classes=10

#model 만들기.
fashion_model=Sequential()
fashion_model.add(Conv2D(32,kernel_size=(3,3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2,2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64,(3,3),activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2,2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128,(3,3),activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2,2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Flatten())
fashion_model.add(Dense(128,activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.3))#왜 0.3인가?
fashion_model.add(Dense(num_classes, activation='softmax'))
#The last layer is a Dense layer that has a softmax activation function with 10 units
# multi-class classification problem을 해결한다.

#adam optimizer로 최적화, multi-class classification을 다루기 위해 categorical cross entropy 사용.
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

#20 번의 epoch 훈련, 및 training accuracy and loss를 확인가능하다.
fashion_train_dropout=fashion_model.fit(train_X,train_label,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X,valid_label))
#각각의 layer의 parameter(weights and biases) 볼 수 있고 total parameter도 보여준다.
fashion_model.summary()

fashion_model.save("fashion_model_dropout.h5py")
#same the model after every epoch so will not to start the training from the beginning

#그러나 값들이 overfitting하였다. training accuracy는 높은데 validation accuracy는 낮기 때문이다.(그래서 dropout사용함)
#fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

#plot 정확도 + 손실(trainingset과 validation data 사이)

accuracy = fashion_train_dropout.history['acc']
val_accuracy = fashion_train_dropout.history['val_acc']
loss = fashion_train_dropout.history['loss']
val_loss = fashion_train_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()