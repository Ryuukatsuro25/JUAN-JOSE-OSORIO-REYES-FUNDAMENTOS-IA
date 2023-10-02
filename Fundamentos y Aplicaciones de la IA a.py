import numpy as np

# Definir la función de activación (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación (para backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

# Datos de entrenamiento
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inicializar pesos y sesgos de manera aleatoria
input_size = 2
hidden_size = 4
output_size = 1

np.random.seed(0)
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
bias_output = np.zeros((1, output_size))

# Hiperparámetros
learning_rate = 0.1
epochs = 10000

# Entrenamiento de la red neuronal
for epoch in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output)

    # Calcular el error
    error = y - predicted_output

    # Backpropagation
    d_output = error * sigmoid_derivative(predicted_output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Actualizar pesos y sesgos
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        print(f'Error en la época {epoch}: {np.mean(np.abs(error))}')

# Resultados finales
print("Resultado después del entrenamiento:")
print(predicted_output)
