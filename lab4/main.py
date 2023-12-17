import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

X = np.arange(-20, 20, 0.1)
y = X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=500, verbose=0)
y_pred = model.predict(X_test)
plt.scatter(X, y, label='Actual Data', color='b', alpha=0.5)
plt.scatter(X_test, y_pred, label='Predicted Data', color='r', alpha=0.5)
plt.xlabel('X')
plt.ylabel('f(X)')
plt.legend()
plt.show()
