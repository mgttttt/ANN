import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

X = np.arange(-20, 20, 0.1)
y = np.abs(X)

X_train, X_test = [], []
y_train, y_test = [], []
count = 0
for i in range(len(X)):
    if count % 5:
        X_train.append(X[i])
        y_train.append(y[i])
    else:
        X_test.append(X[i])
        y_test.append(y[i])
    count += 1
model = Sequential()
model.add(Dense(8, input_dim=1, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=500, verbose=0)
y_pred = model.predict(X_test)
plt.scatter(X, y, label='Actual Data', color='b', alpha=0.5)
plt.scatter(X_test, y_pred, label='Predicted Data', color='r', alpha=0.5)
plt.xlabel('X')
plt.ylabel('f(X)')
plt.legend()
plt.show()
