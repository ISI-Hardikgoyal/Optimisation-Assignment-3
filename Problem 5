import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Rosenbrock function
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Define the gradient of the Rosenbrock function
def gradient(x):
    grad_x = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    grad_y = 200 * (x[1] - x[0]**2)
    return np.array([grad_x, grad_y])

# Implement gradient descent
def gradient_descent(x0, step_size, threshold, max_iterations=100000):
    x = x0.copy()
    iterations = 0
    x_history = [x.copy()]
    f_history = [rosenbrock(x)]
    grad_norm_history = []
    
    while iterations < max_iterations:
        grad = gradient(x)
        grad_norm = np.linalg.norm(grad)
        grad_norm_history.append(grad_norm)
        
        if grad_norm < threshold:
            break
            
        x = x - step_size * grad
        x_history.append(x.copy())
        f_history.append(rosenbrock(x))
        iterations += 1
        
        # Print progress every 10000 iterations
        if iterations % 10000 == 0:
            print(f"Iteration {iterations}, Function value: {f_history[-1]:.6f}, Gradient norm: {grad_norm:.6f}")
    
    return np.array(x_history), np.array(f_history), np.array(grad_norm_history), iterations

# Parameters
x0 = np.array([-1, 1])
step_size = 0.001
threshold = 1e-6

# Run gradient descent
print("Starting gradient descent...")
x_history, f_history, grad_norm_history, iterations = gradient_descent(x0, step_size, threshold)

# Print results
print(f"Total iterations: {iterations}")
print(f"Final point: [{x_history[-1, 0]:.6f}, {x_history[-1, 1]:.6f}]")
print(f"Final function value: {f_history[-1]:.6f}")
print(f"Final gradient norm: {grad_norm_history[-1]:.6f}")

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
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Calculate function values for the contour plot
for i in range(len(x)):
    for j in range(len(y)):
        Z[j, i] = rosenbrock(np.array([X[j, i], Y[j, i]]))

# Plot the 2D contour with descent path
plt.figure(figsize=(10, 8))
contour = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
plt.colorbar(contour, label='Function Value (log scale)')
plt.plot(x_history[:, 0], x_history[:, 1], 'r.-', linewidth=1, markersize=2, alpha=0.7)
plt.plot(x_history[0, 0], x_history[0, 1], 'go', markersize=10, label='Start')
plt.plot(x_history[-1, 0], x_history[-1, 1], 'mo', markersize=10, label='End')
plt.plot(1, 1, 'b*', markersize=10, label='Global Minimum (1,1)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rosenbrock Function: Gradient Descent Path')
plt.legend()
plt.grid(True)

# Create 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Limit the Z range to make the plot more readable
Z_plot = np.minimum(Z, 100)  # Cap at 100 for better visualization

surf = ax.plot_surface(X, Y, Z_plot, cmap='viridis', alpha=0.8, edgecolor='none')
ax.plot(x_history[:, 0], x_history[:, 1], [rosenbrock(x) for x in x_history], 'r.-', linewidth=2, markersize=0)
ax.scatter(x_history[0, 0], x_history[0, 1], rosenbrock(x_history[0]), color='green', s=100, label='Start')
ax.scatter(x_history[-1, 0], x_history[-1, 1], rosenbrock(x_history[-1]), color='magenta', s=100, label='End')
ax.scatter(1, 1, 0, color='blue', s=100, marker='*', label='Global Minimum (1,1)')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X,Y)')
ax.set_title('Rosenbrock Function Surface with Gradient Descent Path')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.legend()

plt.show()
