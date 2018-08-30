from keras.models import Sequential
from keras.layers import Dense,Flatten
import tensorflow as tf
import numpy as np

x_train=[0]*100
y_train=[0]*100
for i in range(100):
	x_train[i]=i
	if(i>50):
		y_train[i]=1
	else:
		y_train[i]=0	
#print(x_train,y_train)
x_test=([1,100,50,45,55,30,78])
model = Sequential()
#model.add(Flatten(input_shape=(3,100)))
model.add(Dense(units=100 ,activation='relu', input_dim=1))
model.add(Dense(units=50 ,activation='relu'))
model.add(Dense(units=50 ,activation='relu'))
model.add(Dense(units=1 ,activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=100)
predict = model.predict(x_test)

print(predict)