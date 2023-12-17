import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math


class Neuron:
    def __init__(self, n):
        self.weights = [random.randint(1, 200) / 1000 for _ in range(n)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def predict(self, x: list) -> float:
        return self.sigmoid(x[0] * self.weights[0] + x[1] * self.weights[1] + x[2] * self.weights[2])


class NeuralNetwork:
    def __init__(self, n):
        self.neurons = [Neuron(3) for _ in range(n)]
        self.a = 0.00005

    def predict(self, x: list):
        return self.neurons[0].predict(x)

    def fit_1(self, x: list, y: list, x_test: list, y_test: list):
        mse = 0
        length = len(x)
        for i in range(length):
            y_pred = self.predict(x[i])
            self.neurons[0].weights[0] -= self.a * (y_pred - y[i]) * x[i][0]
            self.neurons[0].weights[1] -= self.a * (y_pred - y[i]) * x[i][1]
            self.neurons[0].weights[2] -= self.a * (y_pred - y[i]) * x[i][2]
        length_2 = len(x_test)
        for i in range(length_2):
            y_pred = self.predict(x_test[i])
            mse += ((y_pred - y_test[i]) ** 2) / length_2
        print(mse)
        return [self.neurons[0].weights[0], self.neurons[0].weights[1], self.neurons[0].weights[2]]


im = Image.open('../image.jpg')
a = np.asarray(im)
x_train = []
y_train = []
x_test = []
y_test = []
i = 0
red_fragments = list(a[400:450, 350:450])
non_red_fragments = list(a[900:1000, 1500:1550])
for red, non_red in zip(red_fragments, non_red_fragments):
    for y, n in zip(red, non_red):
        #print(y, n)
        if (i % 5):
            #print(y[0])
            x_train.append(y)
            y_train.append(1)
            x_train.append(n)
            y_train.append(0)
        else:
            x_test.append(y)
            y_test.append(1)
            x_train.append(n)
            y_train.append(0)
        i += 1
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
mas = []
for element in a:
    mas.append(element)
mas = np.array(mas)

nn = NeuralNetwork(1)
print(nn.fit_1(x_train, y_train, x_test, y_test))
res = []
for i in range(len(mas)):
    lst = []
    for j in range(len(mas[i])):
        lst.append(100 * round(nn.predict(mas[i][j])))
    res.append(np.array(lst))
print(res)
plt.figure(figsize=(15.,10.))
plt.imshow(Image.fromarray(np.array(res, dtype='uint8'), mode="L"))
plt.show()
