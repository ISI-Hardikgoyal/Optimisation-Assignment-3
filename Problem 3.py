import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Set random seed for reproducibility
np.random.seed(42)

# Define the quadratic convex function: f(x) = x^T A x + b^T x
def quadratic_function(x, A, b):
    return x.T @ A @ x + b.T @ x

# Define the gradient of the quadratic function: ∇f(x) = 2Ax + b
def gradient(x, A, b):
    return 2 * A @ x + b

# Gradient descent implementation
def gradient_descent(x_init, A, b, eta, epsilon, max_iter=10000):
    x = x_init.copy()
    n_iter = 0
    converged = False
    
    # Store values for plotting
    xs = [x.copy()]
    losses = [quadratic_function(x, A, b)]
    grad_norms = []
    
    while not converged and n_iter < max_iter:
        # Calculate gradient
        grad = gradient(x, A, b)
        
        # Update x
        x = x - eta * grad
        
        # Store for plotting
        xs.append(x.copy())
        current_loss = quadratic_function(x, A, b)
        losses.append(current_loss)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)
        
        if grad_norm < epsilon:
            converged = True
            
        n_iter += 1
    
    return x, n_iter, xs, losses, grad_norms, converged

# Main execution
def main():
    # Define the parameters for Problem 3: Quadratic Convex Function
    # A = [[2, 0], [0, 4]]
    A = np.array([[2, 0], [0, 4]])
    # b = [-4, -8]
    b = np.array([-4, -8])
    
    # Initial point: x_0 = [1, 1]
    x_init = np.array([1, 1])
    
    # Step size: η = 0.1
    eta = 0.1
    
    # Threshold: ε = 10^(-6)
    epsilon = 1e-6
    
    # Run gradient descent
    x_final, n_iter, xs, losses, grad_norms, converged = gradient_descent(x_init, A, b, eta, epsilon)
    
    # Convert lists to arrays for easier plotting
    xs_array = np.array(xs)
    
    # The analytic solution for this quadratic function is x* = -A^(-1)b/2
    A_inv = np.linalg.inv(A)
    analytic_solution = -0.5 * A_inv @ b
    
    # Print results
    print(f"{'Convergence status:':<25} {'Converged' if converged else 'Maximum iterations reached'}")
    print(f"{'Number of iterations:':<25} {n_iter}")
    print(f"{'Final loss value:':<25} {losses[-1]:.10f}")
    print(f"{'Final gradient norm:':<25} {grad_norms[-1] if grad_norms else 'N/A'}")
    print("\nFinal solution:")
    print(f"{'x1:':<25} {x_final[0]:.10f}")
    print(f"{'x2:':<25} {x_final[1]:.10f}")
    
    print("\nAnalytic solution for comparison:")
    print(f"{'x1*:':<25} {analytic_solution[0]:.10f}")
    print(f"{'x2*:':<25} {analytic_solution[1]:.10f}")
    
    # Calculate distance from analytic solution
    distance = np.linalg.norm(x_final - analytic_solution)
    print(f"\n{'Distance from analytic solution:':<25} {distance:.10f}")
    
    # Plot loss vs iterations
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(n_iter + 1), losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iterations')
    plt.grid(True)
    
    # Plot with log scale for y-axis to better see convergence
    plt.subplot(1, 2, 2)
    plt.semilogy(range(n_iter + 1), losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss vs. Iterations (Log Scale)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('quadratic_loss_convergence.png')
    
    # Plot the gradient norm
    plt.figure(figsize=(10, 5))
    plt.semilogy(range(n_iter), grad_norms)
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm (log scale)')
    plt.title('Gradient Norm vs. Iterations')
    plt.grid(True)
    plt.savefig('quadratic_gradient_norm.png')
    
    # Create a meshgrid for the loss surface
    x1_range = np.linspace(-1, 2, 100)
    x2_range = np.linspace(-1, 2, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.zeros_like(X1)
    
    # Calculate function value for each point in the grid
    for i in range(len(x1_range)):
        for j in range(len(x2_range)):
            x = np.array([X1[i, j], X2[i, j]])
            Z[i, j] = quadratic_function(x, A, b)
    
    # Plot the loss surface in 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm, alpha=0.8)
    
    # Plot the descent path in 3D
    ax.plot(xs_array[:, 0], xs_array[:, 1], [quadratic_function(x, A, b) for x in xs_array], 
            'r-o', markersize=3, label='Gradient Descent Path')
    
    # Mark the starting and ending points
    ax.scatter(x_init[0], x_init[1], quadratic_function(x_init, A, b), 
               color='green', s=100, label='Initial Point')
    ax.scatter(x_final[0], x_final[1], quadratic_function(x_final, A, b), 
               color='blue', s=100, label='Final Point')
    ax.scatter(analytic_solution[0], analytic_solution[1], 
               quadratic_function(analytic_solution, A, b), 
               color='purple', s=100, label='Analytic Solution')
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    ax.set_title('3D Visualization of Gradient Descent on Quadratic Function')
    ax.legend()
    
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('quadratic_descent_path_3d.png')
    
    # 2D contour plot with descent path
    plt.figure(figsize=(10, 8))
    contour = plt.contour(X1, X2, Z, 20, cmap='viridis')
    plt.plot(xs_array[:, 0], xs_array[:, 1], 'r-o', markersize=3, label='Gradient Descent Path')
    plt.scatter(x_init[0], x_init[1], color='green', s=100, label='Initial Point')
    plt.scatter(x_final[0], x_final[1], color='blue', s=100, label='Final Point')
    plt.scatter(analytic_solution[0], analytic_solution[1], color='purple', s=100, label='Analytic Solution')
    plt.colorbar(contour)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot with Gradient Descent Path')
    plt.legend()
    plt.grid(True)
    plt.savefig('quadratic_descent_path_2d.png')
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()
