import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

x = np.linspace(0, 8 * np.pi, 1000)
y = np.sin(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Dense(16, input_dim=1, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train, y_train, epochs=300, batch_size=10)

y_pred = model.predict(X_test)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label='sin(x)')
plt.scatter(X_test, y_pred, label='Предсказанная sin(x)', linestyle='--')
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot([i for i in range(300)], history.history['loss'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
