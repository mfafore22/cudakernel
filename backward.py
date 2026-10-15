import numpy as np


def softmax(Z):
    # Subtract the row max so exp() can't overflow.
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def relu_derivative(Z):
    return (Z > 0).astype(Z.dtype)


def backward_batch(X, Z1, A1, Z2, y_true, W1, W2):
    batch_size = X.shape[0]

    probs = softmax(Z2)
    grad_Z2 = probs.copy()
    grad_Z2[np.arange(batch_size), y_true] -= 1
    grad_Z2 /= batch_size

    grad_W2 = A1.T @ grad_Z2
    grad_b2 = np.sum(grad_Z2, axis=0)

    grad_A1 = grad_Z2 @ W2.T
    grad_Z1 = grad_A1 * relu_derivative(Z1)

    grad_W1 = X.T @ grad_Z1
    grad_b1 = np.sum(grad_Z1, axis=0)

    return grad_W1, grad_b1, grad_W2, grad_b2