import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

radius = 1

#angles = np.linspace(0, np.pi, 1000)
angles2 = np.linspace(0, 2 * np.pi, 1000)
x = radius * np.cos(angles2)
y = radius * np.sin(angles2)
x = list(x)
y = list(y)
x2 = []
for i in range(len(x)):
    if y[i] < 0:
        x2.append([x[i], -1])
    else:
        x2.append([x[i], 1])
#data = np.column_stack((x, y))
X_train, X_test, y_train, y_test = train_test_split(x2, y, test_size=0.2, random_state=42)
grah = []
for i in range(len(X_test)):
    grah.append(X_test[i][0])
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=50, batch_size=32)

predicted_data = model.predict(X_test)

plt.figure(figsize=(8, 8))
plt.plot(x, y, label='Исходная окружность', color='blue')

plt.scatter(grah, predicted_data, label='Предсказания модели', color='red', s=1)

plt.title('Аппроксимация окружности с помощью Keras')
plt.xlabel('Координата X')
plt.ylabel('Координата Y')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.show()
