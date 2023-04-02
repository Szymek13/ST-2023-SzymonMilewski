import numpy as np

#ZADANIE1a

class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                self.W = self.W + self.lr * e * x

X = np.array([[0,0], [0,1], [1,0], [1,1]])
d = np.array([0,0,0,1])

perceptron = Perceptron(input_size=2)
perceptron.fit(X, d)
print("Zadanie 1 [AND]")
print(perceptron.W)

#ZADANIE1b

class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                self.W = self.W + self.lr * e * x

X = np.array([[0], [1]])
d = np.array([1,0])

perceptron = Perceptron(input_size=1)
perceptron.fit(X, d)
print("Zadanie 1 [NOT]")
print(perceptron.W)

#ZADANIE2

class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                self.W = self.W + self.lr * e * x

X = np.array([[1,0], [1,1], [0,1], [0,0]])
d = np.array([1,0,0,0])

perceptron = Perceptron(input_size=2)
perceptron.fit(X, d)
print("Zadanie 2")
print(perceptron.W)

#ZADANIE3

class Perceptron:
    def __init__(self, input_size, output_size, lr=1, epochs=10):
        self.W = np.random.rand(input_size+1, output_size) * 2 - 1
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        return x * (1 - x)

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                delta = e * self.activation_derivative(y)
                self.W = self.W + self.lr * np.outer(x, delta)

class XORPerceptron:
    def __init__(self):
        self.hidden_layer = Perceptron(input_size=2, output_size=2)
        self.output_layer = Perceptron(input_size=2, output_size=1)

    def predict(self, x):
        h = self.hidden_layer.predict(x)
        o = self.output_layer.predict(h)
        return np.round(o)

    def fit(self, X, d):
        for epoch in range(self.hidden_layer.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                h = self.hidden_layer.predict(x)
                o = self.output_layer.predict(h)
                e = d[i] - o
                delta = e * self.output_layer.activation_derivative(o)
                delta_h = self.hidden_layer.activation_derivative(h) * delta.dot(self.output_layer.W.T)
                x = np.insert(x, 0, 1)
                self.output_layer.W = self.output_layer.W + self.output_layer.lr * np.outer(h, delta)
                self.hidden_layer.W = self.hidden_layer.W + self.hidden_layer.lr * np.outer(x, delta_h)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
d = np.array([0, 1, 1, 0])

perceptron = XORPerceptron()
perceptron.fit(X, d)
print("Zadanie 3")
print(perceptron.predict(X))










