import numpy as np
import matplotlib.pyplot as plt
# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n = 100
p = 2
eta = 0.01
epsilon = 1e-6
max_iters = 10000

# Generate data
x_raw = np.random.normal(1, 2, size=(n,))
X = np.column_stack((np.ones(n), x_raw))  # add intercept term
y = np.random.normal(2 + 3 * x_raw, 5)

# Loss function
def compute_loss(X, y, beta):
    residuals = y - X @ beta
    return (1 / (2 * len(y))) * np.sum(residuals ** 2)

# Gradient
def compute_gradient(X, y, beta):
    return -(1 / len(y)) * X.T @ (y - X @ beta)

# Gradient Descent
def gradient_descent(X, y, beta_init, eta, epsilon, max_iters):
    beta = beta_init.copy()
    loss_history = []
    beta_path = [beta.copy()]

    for i in range(max_iters):
        grad = compute_gradient(X, y, beta)
        grad_norm = np.linalg.norm(grad)
        loss = compute_loss(X, y, beta)

        loss_history.append(loss)
        if grad_norm < epsilon:
            print(f"Converged in {i} iterations.")
            break

        beta -= eta * grad
        beta_path.append(beta.copy())

    return beta, loss_history, np.array(beta_path)

# Run gradient descent
beta_init = np.zeros(p)
beta_final, loss_history, beta_path = gradient_descent(X, y, beta_init, eta, epsilon, max_iters)

# Results
print(f"Final beta: {beta_final}")

# Plot loss vs iterations
plt.plot(loss_history)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs. Iterations")
plt.grid(True)
plt.show()

# Optional: 2D path visualization
plt.plot(beta_path[:, 0], beta_path[:, 1], marker='o')
plt.xlabel("Beta 0")
plt.ylabel("Beta 1")
plt.title("Gradient Descent Path")
plt.grid(True)
plt.show()

