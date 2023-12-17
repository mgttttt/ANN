import random


class Neuron:
    def __init__(self, n):
        self.weights = [random.randint(1, 200) / 1000 for _ in range(n)]

    def predict(self, x: list) -> float:
        return sum([x[i] * self.weights[i] for i in range(len(self.weights))])


class NeuralNetwork:
    def __init__(self, n):
        self.neurons = [Neuron(6) for _ in range(n)]
        self.a = 0.00005
        self.e = 10

    def predict(self, x: list):
        return [self.neurons[i].predict(x) for i in range(len(self.neurons))]

    def fit_1(self, x: list, y: list, x_test: list, y_test: list):
        mse = 0
        length = len(x)
        #print(self.neurons[0].weights)
        for i in range(length):
            y_pred = self.predict(x[i])
            for k in range(len(y_pred)):
                for j in range(len(self.neurons[0].weights)):
                    self.neurons[k].weights[j] -= self.a * (y_pred[k] - y[i][k]) * x[i][j]
            #mse += ((y_pred - y[i]) ** 2) / length
            #self.neurons[0].weights[0] -= self.a * (y_pred - y[i]) * x[i][0]
            #self.neurons[0].weights[1] -= self.a * (y_pred - y[i]) * x[i][1]
            #print(self.neurons[0].weights[0])
        length_2 = len(x_test)
        for i in range(length_2):
            y_pred = self.predict(x_test[i])
            for k in range(len(y_pred)):
                mse += ((y_pred[k] - y_test[i][k]) ** 2) / length_2
        print(mse)
        return [self.neurons[i].weights for i in range(len(self.neurons))]


f = open('../2lab_data.csv', 'r')
x = []
y = []
x_test = []
y_test = []

count = 0
for line in f.read().split('\n'):
    lst = line.split(',')
    if count % 5:
        x.append([int(lst[i]) for i in range(6)])
        y.append([int(lst[6]), int(lst[7]), int(lst[8])])
    else:
        x_test.append([int(lst[i]) for i in range(6)])
        y_test.append([int(lst[6]), int(lst[7]), int(lst[8])])
    count += 1

nn = NeuralNetwork(3)
print(nn.fit_1(x, y, x_test, y_test))
