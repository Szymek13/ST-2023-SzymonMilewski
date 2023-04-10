import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#Zadanie 4

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)
synapse_0 = 2 * np.random.random((2, 4)) - 1
synapse_1 = 2 * np.random.random((4, 1)) - 1

for j in range(60000):

    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))

    layer_2_error = y - layer_2

    if (j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(layer_2_error))))

    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(synapse_1.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    synapse_1 += layer_1.T.dot(layer_2_delta)
    synapse_0 += layer_0.T.dot(layer_1_delta)

print("Wynik po nauczeniu:")
print(layer_2)

#Zadanie 5

input_layer_size = 2
hidden_layer_size = 3
output_layer_size = 2

weights1 = np.array([
    [0.1, -0.2],
    [0, 0.2],
    [0.3, -0.4]
])

weights2 = np.array([
    [-0.4, 0.1, 0.6],
    [-0.1, -0.2, 0.2]
])

training_inputs = np.array([[0.2, 0.3], [0.4, 0.5], [0.1, 0.7]])
training_outputs = np.array([[0, 1], [1, 0], [0, 1]])

learning_rate = 0.1
for i in range(10000):
    # propagacja w przód
    hidden_layer_outputs = sigmoid(np.dot(training_inputs, weights1.T))
    output_layer_outputs = sigmoid(np.dot(hidden_layer_outputs, weights2.T))

    # propagacja wsteczna błędu
    output_layer_errors = (training_outputs - output_layer_outputs) * output_layer_outputs * (1 - output_layer_outputs)
    hidden_layer_errors = np.dot(output_layer_errors, weights2) * hidden_layer_outputs * (1 - hidden_layer_outputs)

    # aktualizacja wag
    weights2 += learning_rate * np.dot(output_layer_errors.T, hidden_layer_outputs)
    weights1 += learning_rate * np.dot(hidden_layer_errors.T, training_inputs)

test_input = np.array([[0.6,0.1],[0.2,0.3]])
hidden_layer_output = sigmoid(np.dot(test_input, weights1.T))
output_layer_output = sigmoid(np.dot(hidden_layer_output, weights2.T))


print("Nowe wagi dla warstwy ukrytej:")
print(weights1,"\n")
print("Nowe wagi dla warstwy wyjściowej:")
print(weights2,"\n")

print("Wynik dla testowego wejścia", test_input, "wynosi:", np.round(output_layer_output[0]))