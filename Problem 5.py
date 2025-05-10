import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Set random seed for reproducibility
np.random.seed(42)

# Define the Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
def rosenbrock_function(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Define the gradient of the Rosenbrock function
def rosenbrock_gradient(x):
    grad_x = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    grad_y = 200 * (x[1] - x[0]**2)
    return np.array([grad_x, grad_y])

# Gradient descent implementation
def gradient_descent(x_init, grad_func, func, eta, epsilon, max_iter=100000):
    x = x_init.copy()
    n_iter = 0
    converged = False
    
    # Store values for plotting
    xs = [x.copy()]
    losses = [func(x)]
    grad_norms = []
    
    while not converged and n_iter < max_iter:
        # Calculate gradient
        grad = grad_func(x)
        
        # Update x
        x = x - eta * grad
        
        # Store for plotting
        xs.append(x.copy())
        current_loss = func(x)
        losses.append(current_loss)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)
        
        if grad_norm < epsilon:
            converged = True
            
        n_iter += 1
        
        # Print progress every 1000 iterations
        if n_iter % 1000 == 0:
            print(f"Iteration {n_iter}: Loss = {current_loss:.8f}, Gradient norm = {grad_norm:.8f}")
    
    return x, n_iter, xs, losses, grad_norms, converged

# Main execution
def main():
    # Define the parameters for Problem 5: Rosenbrock Function
    
    # Initial point: [x₀, y₀] = [-1, 1]
    x_init = np.array([-1, 1])
    
    # Step size: η = 0.001
    eta = 0.001
    
    # Threshold: ε = 10^(-6)
    epsilon = 1e-6
    
    # Known global minimum of the Rosenbrock function
    global_min = np.array([1, 1])
    
    print("Starting gradient descent for Rosenbrock function...")
    print(f"Initial point: [{x_init[0]}, {x_init[1]}]")
    print(f"Step size: {eta}")
    print(f"Convergence threshold: {epsilon}")
    print(f"Known global minimum: [{global_min[0]}, {global_min[1]}]")
    print("\nOptimization progress:")
    
    # Run gradient descent
    x_final, n_iter, xs, losses, grad_norms, converged = gradient_descent(
        x_init, 
        rosenbrock_gradient, 
        rosenbrock_function, 
        eta, 
        epsilon
    )
    
    # Convert lists to arrays for easier plotting
    xs_array = np.array(xs)
    
    # Print results
    print("\nResults:")
    print(f"{'Convergence status:':<25} {'Converged' if converged else 'Maximum iterations reached'}")
    print(f"{'Number of iterations:':<25} {n_iter}")
    print(f"{'Final loss value:':<25} {losses[-1]:.10f}")
    print(f"{'Final gradient norm:':<25} {grad_norms[-1] if grad_norms else 'N/A'}")
    print("\nFinal solution:")
    print(f"{'x:':<25} {x_final[0]:.10f}")
    print(f"{'y:':<25} {x_final[1]:.10f}")
    
    print("\nDistance from global minimum:")
    distance = np.linalg.norm(x_final - global_min)
    print(f"{'Distance:':<25} {distance:.10f}")
    
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
    plt.savefig('rosenbrock_loss_convergence.png')
    
    # Plot the gradient norm
    plt.figure(figsize=(10, 5))
    plt.semilogy(range(n_iter), grad_norms)
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm (log scale)')
    plt.title('Gradient Norm vs. Iterations')
    plt.grid(True)
    plt.savefig('rosenbrock_gradient_norm.png')
    
    # Create a meshgrid for the contour plot
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    # Calculate function value for each point in the grid
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            Z[j, i] = rosenbrock_function(np.array([X[j, i], Y[j, i]]))
    
    # Plot the loss surface in 3D with reduced data points for better performance
    stride = 5  # Plot every 5th point for performance
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride], 
                          cmap=cm.coolwarm, alpha=0.7, linewidth=0)
    
    # Plot the descent path in 3D
    # Only plot a subset of points to avoid cluttering the graph
    num_points = min(500, len(xs_array))
    indices = np.linspace(0, len(xs_array) - 1, num_points, dtype=int)
    
    ax.plot(xs_array[indices, 0], xs_array[indices, 1], 
            [rosenbrock_function(x) for x in xs_array[indices]], 
            'r-', markersize=2, label='Gradient Descent Path')
    
    # Mark the starting and ending points
    ax.scatter(x_init[0], x_init[1], rosenbrock_function(x_init), 
               color='green', s=100, label='Initial Point')
    ax.scatter(x_final[0], x_final[1], rosenbrock_function(x_final), 
               color='blue', s=100, label='Final Point')
    ax.scatter(global_min[0], global_min[1], rosenbrock_function(global_min), 
               color='purple', s=100, label='Global Minimum')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title('3D Visualization of Gradient Descent on Rosenbrock Function')
    ax.legend()
    
    plt.savefig('rosenbrock_descent_path_3d.png')
    
    # 2D contour plot with descent path
    plt.figure(figsize=(10, 8))
    
    # Plot contours of the Rosenbrock function
    levels = np.logspace(-1, 3, 30)  # Log scale for levels to better visualize the valley
    contour = plt.contour(X, Y, Z, levels=levels, cmap='viridis')
    plt.colorbar(contour, label='f(x,y)')
    
    # Plot the descent path - use different sampling for clarity
    path_density = max(1, n_iter // 1000)  # Take at most 1000 points to avoid cluttering
    plt.plot(xs_array[::path_density, 0], xs_array[::path_density, 1], 
             'r-', linewidth=1, label='Gradient Descent Path')
    
    # Mark important points
    plt.scatter(x_init[0], x_init[1], color='green', s=100, label='Initial Point')
    plt.scatter(x_final[0], x_final[1], color='blue', s=100, label='Final Point')
    plt.scatter(global_min[0], global_min[1], color='purple', s=100, label='Global Minimum')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour Plot of Rosenbrock Function with Gradient Descent Path')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Equal scaling on both axes
    plt.savefig('rosenbrock_descent_path_2d.png')
    
    # Plot a zoomed-in version near the minimum to see the final convergence
    plt.figure(figsize=(10, 8))
    
    # Create a zoomed meshgrid
    x_zoom = np.linspace(0.5, 1.5, 100)
    y_zoom = np.linspace(0.5, 1.5, 100)
    X_zoom, Y_zoom = np.meshgrid(x_zoom, y_zoom)
    Z_zoom = np.zeros_like(X_zoom)
    
    # Calculate function value for each point in the zoomed grid
    for i in range(len(x_zoom)):
        for j in range(len(y_zoom)):
            Z_zoom[j, i] = rosenbrock_function(np.array([X_zoom[j, i], Y_zoom[j, i]]))
    
    # Plot zoomed contours
    zoom_levels = np.logspace(-4, 0, 20)
    contour_zoom = plt.contour(X_zoom, Y_zoom, Z_zoom, levels=zoom_levels, cmap='viridis')
    plt.colorbar(contour_zoom, label='f(x,y)')
    
    # Filter the path points to only show those in the zoomed area
    in_zoom_area = np.logical_and.reduce([
        xs_array[:, 0] >= x_zoom.min(), 
        xs_array[:, 0] <= x_zoom.max(),
        xs_array[:, 1] >= y_zoom.min(), 
        xs_array[:, 1] <= y_zoom.max()
    ])
    
    zoom_path = xs_array[in_zoom_area]
    
    # Plot the filtered path
    if len(zoom_path) > 0:
        plt.plot(zoom_path[:, 0], zoom_path[:, 1], 'r-o', markersize=2, label='Gradient Descent Path')
    
    # Mark important points if they're in the zoom area
    if 0.5 <= x_final[0] <= 1.5 and 0.5 <= x_final[1] <= 1.5:
        plt.scatter(x_final[0], x_final[1], color='blue', s=100, label='Final Point')
    
    plt.scatter(global_min[0], global_min[1], color='purple', s=100, label='Global Minimum')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Zoomed Contour Plot Near the Minimum')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('rosenbrock_descent_path_zoomed.png')
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()
