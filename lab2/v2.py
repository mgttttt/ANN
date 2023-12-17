import random

#random.randint(1, 200) / 1000 for _ in range(n)
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

    def fit_1(self, x: list, y: list):
        mse = 0
        length = len(x)
        print(self.neurons[0].weights)
        for i in range(length):
            y_pred = self.predict(x[i])
            mse += ((y_pred - y[i]) ** 2) / length
            self.neurons[0].weights[0] -= self.a * (y_pred - y[i]) * x[i][0]
            self.neurons[0].weights[1] -= self.a * (y_pred - y[i]) * x[i][1]
            #print(self.neurons[0].weights[0])
        print(mse)
        return [self.neurons[0].weights[0], self.neurons[0].weights[1]]


f = open('../data.csv', 'r')
x = []
y = []

for line in f.read().split('\n'):
    lst = line.split(';')
    x.append([int(lst[0]), int(lst[1])])
    y.append(int(lst[2]))

nn = NeuralNetwork(1)
print(nn.fit_1(x, y))
