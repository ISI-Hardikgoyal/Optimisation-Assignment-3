import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the quadratic function
def quadratic_function(x):
    A = np.array([[2, 0], [0, 4]])
    b = np.array([-4, -8])
    return x.T @ A @ x + b.T @ x

# Define the gradient of the function
def gradient(x):
    A = np.array([[2, 0], [0, 4]])
    b = np.array([-4, -8])
    return 2 * A @ x + b

# Implement gradient descent
def gradient_descent(x0, step_size, threshold, max_iterations=1000):
    x = x0.copy()
    iterations = 0
    x_history = [x.copy()]
    f_history = [quadratic_function(x)]
    grad_norm_history = []
    
    while iterations < max_iterations:
        grad = gradient(x)
        grad_norm = np.linalg.norm(grad)
        grad_norm_history.append(grad_norm)
        
        if grad_norm < threshold:
            break
            
        x = x - step_size * grad
        x_history.append(x.copy())
        f_history.append(quadratic_function(x))
        iterations += 1
    
    return np.array(x_history), np.array(f_history), np.array(grad_norm_history), iterations

# Parameters
x0 = np.array([1, 1])
step_size = 0.1
threshold = 1e-6

# Run gradient descent
x_history, f_history, grad_norm_history, iterations = gradient_descent(x0, step_size, threshold)

# Print results
print(f"Total iterations: {iterations}")
print(f"Final point: {x_history[-1]}")
print(f"Final function value: {f_history[-1]}")
print(f"Final gradient norm: {grad_norm_history[-1]}")

# Plot the loss vs. iterations
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(len(f_history)), f_history)
plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.title('Loss vs. Iterations')
plt.grid(True)

# Plot the gradient norm vs. iterations
plt.subplot(1, 2, 2)
plt.plot(range(len(grad_norm_history)), grad_norm_history)
plt.xlabel('Iterations')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm vs. Iterations')
plt.grid(True)
plt.tight_layout()

# Create a meshgrid for the contour plot
x1 = np.linspace(-1, 2, 100)
x2 = np.linspace(-1, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros_like(X1)

# Calculate function values for the contour plot
for i in range(len(x1)):
    for j in range(len(x2)):
        Z[j, i] = quadratic_function(np.array([X1[j, i], X2[j, i]]))

# Plot the 2D contour with descent path
plt.figure(figsize=(10, 8))
contour = plt.contour(X1, X2, Z, levels=20, cmap='viridis')
plt.plot(x_history[:, 0], x_history[:, 1], 'r.-', linewidth=2, markersize=8)
plt.plot(x_history[0, 0], x_history[0, 1], 'go', markersize=10, label='Start')
plt.plot(x_history[-1, 0], x_history[-1, 1], 'mo', markersize=10, label='End')
plt.colorbar(contour)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gradient Descent Path')
plt.legend()
plt.grid(True)

# Calculate the analytical solution for verification
A = np.array([[2, 0], [0, 4]])
b = np.array([-4, -8])
x_analytical = -0.5 * np.linalg.inv(A) @ b
print(f"Analytical solution: {x_analytical}")
print(f"Analytical function value: {quadratic_function(x_analytical)}")

plt.show()
