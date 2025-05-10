import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Set random seed for reproducibility
np.random.seed(42)

# Generate dataset
def generate_data(n=100, p=2):
    X = np.random.normal(1, 2, size=(n, p))
    # Add column of 1s for intercept
    X_with_intercept = np.column_stack((np.ones(n), X))
    
    # True parameters (we'll use [1, 2, 3] for simplicity)
    beta_true = np.array([1, 2, 3])
    
    # Generate y values using the specified distribution: y ~ N(2+3x, 5)
    y = 2 + 3 * X[:, 0] + np.random.normal(0, np.sqrt(5), size=n)
    
    return X_with_intercept, y, beta_true

# Linear regression loss function: f(β) = (1/2n) * sum((y_i - x_i^T * β)^2)
def loss_function(X, y, beta):
    n = len(y)
    predictions = X @ beta
    residuals = predictions - y
    return (1 / (2 * n)) * np.sum(residuals ** 2)

# Gradient of the loss function: ∇f(β) = (1/n) * X^T * (X * β - y)
def gradient(X, y, beta):
    n = len(y)
    predictions = X @ beta
    residuals = predictions - y
    return (1 / n) * (X.T @ residuals)

# Gradient descent implementation
def gradient_descent(X, y, beta_init, eta, epsilon, max_iter=10000):
    beta = beta_init.copy()
    n_iter = 0
    converged = False
    
    # Store values for plotting
    betas = [beta.copy()]
    losses = [loss_function(X, y, beta)]
    grad_norms = []
    
    while not converged and n_iter < max_iter:
        # Calculate gradient
        grad = gradient(X, y, beta)
        
        # Update beta
        beta = beta - eta * grad
        
        # Store for plotting
        betas.append(beta.copy())
        current_loss = loss_function(X, y, beta)
        losses.append(current_loss)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)
        
        if grad_norm < epsilon:
            converged = True
            
        n_iter += 1
    
    return beta, n_iter, betas, losses, grad_norms, converged

# Main execution
def main():
    # Generate data
    n, p = 100, 2
    X, y, beta_true = generate_data(n, p)
    
    # Initial point: β₀ = [0, 0]
    # Note: We're adding an intercept, so it becomes [0, 0, 0]
    beta_init = np.zeros(p + 1)
    
    # Step size: η = 0.01
    eta = 0.01
    
    # Threshold: ε = 10⁻⁶
    epsilon = 1e-6
    
    # Run gradient descent
    beta_final, n_iter, betas, losses, grad_norms, converged = gradient_descent(X, y, beta_init, eta, epsilon)
    
    # Convert lists to arrays for easier plotting
    betas_array = np.array(betas)
    
    # Print results
    print(f"{'Convergence status:':<25} {'Converged' if converged else 'Maximum iterations reached'}")
    print(f"{'Number of iterations:':<25} {n_iter}")
    print(f"{'Final loss value:':<25} {losses[-1]:.10f}")
    print(f"{'Final gradient norm:':<25} {grad_norms[-1] if grad_norms else 'N/A'}")
    print("\nFinal solution:")
    print(f"{'β₀ (intercept):':<25} {beta_final[0]:.6f}")
    print(f"{'β₁:':<25} {beta_final[1]:.6f}")
    print(f"{'β₂:':<25} {beta_final[2]:.6f}")
    
    print("\nTrue parameters for comparison:")
    print(f"{'True intercept:':<25} {2.0:.6f}")  # Based on y ~ N(2+3x, 5)
    print(f"{'True β₁:':<25} {3.0:.6f}")         # Coefficient of x
    print(f"{'True β₂:':<25} {0.0:.6f}")         # Should be close to zero
    
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
    plt.savefig('loss_convergence.png')
    
    # Plot the gradient norm
    plt.figure(figsize=(10, 5))
    plt.semilogy(range(n_iter), grad_norms)
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm (log scale)')
    plt.title('Gradient Norm vs. Iterations')
    plt.grid(True)
    plt.savefig('gradient_norm.png')
    
    # Plot the descent path in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid for the loss surface
    b0_range = np.linspace(beta_final[0] - 2, beta_final[0] + 2, 50)
    b1_range = np.linspace(beta_final[1] - 2, beta_final[1] + 2, 50)
    b0_grid, b1_grid = np.meshgrid(b0_range, b1_range)
    
    loss_grid = np.zeros_like(b0_grid)
    for i in range(b0_grid.shape[0]):
        for j in range(b0_grid.shape[1]):
            # Keep beta[2] fixed at final value for visualization
            beta_test = np.array([b0_grid[i, j], b1_grid[i, j], beta_final[2]])
            loss_grid[i, j] = loss_function(X, y, beta_test)
    
    # Plot the loss surface
    surf = ax.plot_surface(b0_grid, b1_grid, loss_grid, alpha=0.7, cmap=cm.coolwarm)
    
    # Plot the descent path (only for β₀ and β₁)
    ax.plot(betas_array[:, 0], betas_array[:, 1], [loss_function(X, y, [b0, b1, beta_final[2]]) 
                                                 for b0, b1 in zip(betas_array[:, 0], betas_array[:, 1])], 
            'r-o', markersize=3, label='Gradient Descent Path')
    
    ax.set_xlabel('β₀')
    ax.set_ylabel('β₁')
    ax.set_zlabel('Loss')
    ax.set_title('3D Visualization of Gradient Descent')
    
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('descent_path_3d.png')
    
    # 2D visualization of the path
    plt.figure(figsize=(10, 8))
    plt.plot(betas_array[:, 0], betas_array[:, 1], 'r-o', markersize=3)
    plt.scatter(beta_final[0], beta_final[1], color='green', s=100, label='Final Point')
    plt.xlabel('β₀')
    plt.ylabel('β₁')
    plt.title('2D Visualization of Gradient Descent Path')
    plt.grid(True)
    plt.legend()
    plt.savefig('descent_path_2d.png')
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()
