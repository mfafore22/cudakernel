def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def initialize_weights(input_size, output_size):
    scale = np.sqrt(6.0 / input_size)
    return (np.random.rand(input_size, output_size) * 2.0 - 1.0) * scale

def initialize_bias(output_size):
    return np.zeros((1, output_size))

def linear_forward(x, weights, bias):
    return x @ weights + bias

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    batch_size = y_pred.shape[0]
    probabilities = softmax(y_pred)
    correct_log_probs = np.log(probabilities[np.arange(batch_size), y_true])
    loss = -np.sum(correct_log_probs) / batch_size
    return loss