import numpy as np
from scipy.optimize import minimize


def sigmoid(y_hat):
    return 1 / (1 + np.exp(-y_hat))


def loss_function(y, y_hat):
    return -y * np.log(sigmoid(y_hat)) - (1 - y) * np.log(1 - sigmoid(y_hat))


def objective_function(params, X, y, J, L, alpha):
    S = params[:J * L].reshape(J, L)
    w = params[J * L:]

    y_hat = np.sum(w[1:] * X, axis=1) + w[0]
    regularization_term = (alpha / len(X)) * np.sum(w[1:] ** 2)

    return np.sum(loss_function(y, y_hat)) + regularization_term


def evaluate_quality(W_i, S, w):
    d_Wi_Sj = np.array([np.linalg.norm(W_i - S_j) for S_j in S])
    y_hat = np.sum(w[1:] * d_Wi_Sj) - w[0]
    y_actual = 1  # Change this to the actual label of W_i
    xi = y_actual - y_hat

    return xi


def learn_shapelets_and_weights(X, y, J, L, alpha):
    I, _ = X.shape
    initial_params = np.random.rand(J * L + J + 1)

    result = minimize(objective_function, initial_params, args=(X, y, J, L, alpha), method='BFGS')

    S = result.x[:J * L].reshape(J, L)
    w = result.x[J * L:]

    return S, w


# Example usage:

J = 5
L = 3 # Set the shapelet length to be equal to the embedding size
alpha = 0.1
X = np.random.rand(100, J)  # 100 samples, each with J features
y = np.random.randint(0, 2, 100)  # 100 binary labels

S, w = learn_shapelets_and_weights(X, y, J, L, alpha)

W_i = np.random.rand(L)  # Example behavior unit candidate
quality = evaluate_quality(W_i, S, w)
print("Quality of the behavior unit candidate:", quality)
