import random


class Neuron:
    def __init__(self, n):
        self.weights = [random.randint(1, 200) / 1000 for _ in range(n)]

    def predict(self, x: list) -> float:
        return x[0] * self.weights[0] + x[1] * self.weights[1]


class NeuralNetwork:
    def __init__(self, n):
        self.neurons = [Neuron(2) for _ in range(n)]
        self.a = 0.00005

    def predict(self, x: list):
        return self.neurons[0].predict(x)

    def fit_1(self, x: list, y: list, x_test: list, y_test: list):
        mse = 0
        length = len(x)
        #print(self.neurons[0].weights)
        for i in range(length):
            y_pred = self.predict(x[i])
            #mse += ((y_pred - y[i]) ** 2) / length
            self.neurons[0].weights[0] -= self.a * (y_pred - y[i]) * x[i][0]
            self.neurons[0].weights[1] -= self.a * (y_pred - y[i]) * x[i][1]
            #print(self.neurons[0].weights[0])
        length_2 = len(x_test)
        for i in range(length_2):
            y_pred = self.predict(x_test[i])
            mse += ((y_pred - y_test[i]) ** 2) / length_2
        print(mse)
        return [self.neurons[0].weights[0], self.neurons[0].weights[1]]

    def fit_2(self, x: list, y: list, x_test: list, y_test: list):
        mse = 0
        length = len(x)
        for i in range(length):
            y_pred = self.predict(x[i])
            #mse += ((y_pred - y[i]) ** 2) / length
            self.neurons[0].weights[0] += (y[i] - y_pred) / sum(self.neurons[0].weights)
            self.neurons[0].weights[1] += (y[i] - y_pred) / sum(self.neurons[0].weights)
            #print(self.neurons[0].weights[0])
        length_2 = len(x_test)
        for i in range(length_2):
            y_pred = self.predict(x_test[i])
            mse += ((y_pred - y_test[i]) ** 2) / length_2
        print(mse)
        return [self.neurons[0].weights[0], self.neurons[0].weights[1]]


f = open('../data.csv', 'r')
x = []
y = []
x_test = []
y_test = []

count = 0
for line in f.read().split('\n'):
    lst = line.split(';')
    if count % 5:
        x.append([int(lst[0]), int(lst[1])])
        y.append(int(lst[2]))
    else:
        x_test.append([int(lst[0]), int(lst[1])])
        y_test.append(int(lst[2]))
    count += 1

nn = NeuralNetwork(1)
nn2 = NeuralNetwork(1)
print(nn.fit_1(x, y, x_test, y_test))
print(nn2.fit_2(x, y, x_test, y_test))
