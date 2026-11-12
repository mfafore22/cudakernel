class NeuralNetwork:
    def __init__(self, input_size, hidden_size,
    output_size):
        self.weights1 = initialize_weights(input_size, hidden_size)
        self.bias1 = initialize_bias(hidden_size)
        self.weights2 = initialize_weights(hidden_size, output_size)
        self.bias2 = initialize_bias(output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        fc1_input = x.reshape(batch_size, -1)
        fc1_output = linear_forward(fc1_input, self.weights1, self.bias1)
        relu_output = relu(fc1_output)
        fc2_output = linear_forward(relu_output, self.weights2, self.bias2)
        return fc2_output, (fc1_input, fc1_output,
        relu_output)

    def backward(self, grad_output, cache):
        x, fc1_output, relu_output = cache
        grad_fc2, grad_weights2, grad_bias2 = linear_backward(grad_output,
        relu_output, self.weights2)
        grad_relu = grad_fc2 * relu_derivative(fc1_output)
        grad_fc1, grad_weights1, grad_bias1 = linear_backward(grad_relu, x,
        self.weights1)
        return grad_weights1, grad_bias1, grad_weights2, grad_bias2

    def update_weights(self, grad_weights1, grad_bias1, grad_weights2,
    grad_bias2, learning_rate):
        self.weights1 -= learning_rate * grad_weights1
        self.bias1 -= learning_rate * grad_bias1
        self.weights2 -= learning_rate * grad_weights2
        self.bias2 -= learning_rate * grad_bias2