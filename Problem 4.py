import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Function to load data from CSV
def load_data(filepath):
    """Load data from CSV file downloaded from Google Sheets"""
    try:
        # Try to load data assuming it's a CSV file
        data = pd.read_csv(filepath)
        # Extract the first column (assuming that's where the data is)
        x_values = data.iloc[:, 0].values
        return x_values
    except Exception as e:
        print(f"Error loading data: {e}")
        print("If you haven't downloaded the data yet, please download it from the Google Sheets link as a CSV file.")
        print("If you're using a different format, please adjust the filepath and loading method.")
        return None

# Define the negative log-likelihood function for normal distribution
def negative_log_likelihood(params, data):
    """
    Calculate negative log-likelihood for normal distribution
    params[0] = mu (mean)
    params[1] = sigma (standard deviation)
    """
    mu, sigma = params
    n = len(data)
    return n * np.log(sigma) + np.sum((data - mu)**2) / (2 * sigma**2)

# Define the gradient of the negative log-likelihood
def gradient(params, data):
    """
    Calculate gradient of negative log-likelihood
    Returns [d/dmu, d/dsigma]
    """
    mu, sigma = params
    n = len(data)
    
    # Gradient with respect to mu
    d_mu = -np.sum(data - mu) / (sigma**2)
    
    # Gradient with respect to sigma
    d_sigma = n / sigma - np.sum((data - mu)**2) / (sigma**3)
    
    return np.array([d_mu, d_sigma])

# Implement gradient descent
def gradient_descent(params_init, data, step_size, threshold, max_iterations=10000):
    """
    Perform gradient descent to minimize negative log-likelihood
    
    Args:
        params_init: Initial parameters [mu, sigma]
        data: Dataset
        step_size: Learning rate
        threshold: Convergence threshold for gradient norm
        max_iterations: Maximum number of iterations
        
    Returns:
        params_history: History of parameters
        nll_history: History of negative log-likelihood values
        grad_norm_history: History of gradient norms
        iterations: Number of iterations performed
    """
    params = params_init.copy()
    iterations = 0
    params_history = [params.copy()]
    nll_history = [negative_log_likelihood(params, data)]
    grad_norm_history = []
    
    while iterations < max_iterations:
        grad = gradient(params, data)
        grad_norm = np.linalg.norm(grad)
        grad_norm_history.append(grad_norm)
        
        if grad_norm < threshold:
            print(f"Converged after {iterations} iterations!")
            break
            
        # Update parameters
        params = params - step_size * grad
        
        # Ensure sigma remains positive
        params[1] = max(params[1], 1e-10)
        
        params_history.append(params.copy())
        nll_history.append(negative_log_likelihood(params, data))
        iterations += 1
        
        # Print progress every 1000 iterations
        if iterations % 1000 == 0:
            print(f"Iteration {iterations}, NLL: {nll_history[-1]:.4f}, Gradient norm: {grad_norm:.6f}")
    
    return np.array(params_history), np.array(nll_history), np.array(grad_norm_history), iterations

def main():
    # Load data - replace with your actual file path
    filepath = "normal_data.csv"  # You'll need to download the Google Sheet as CSV
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"File {filepath} not found!")
        print("Please download the data from the Google Sheets link and save it as 'normal_data.csv' in the same directory as this script.")
        # For demonstration, generate some random data
        print("Generating random data for demonstration...")
        np.random.seed(42)
        data = np.random.normal(loc=5, scale=2, size=100)
    else:
        data = load_data(filepath)
        
    if data is None:
        print("Using randomly generated data for demonstration...")
        np.random.seed(42)
        data = np.random.normal(loc=5, scale=2, size=100)
    
    # Parameters for gradient descent
    params_init = np.array([0, 1])  # Initial mu=0, sigma=1
    step_size = 0.01
    threshold = 1e-5
    
    print(f"Data summary: Mean = {np.mean(data):.4f}, Std = {np.std(data):.4f}")
    print(f"Initial parameters: mu = {params_init[0]}, sigma = {params_init[1]}")
    print(f"Starting gradient descent with step size = {step_size}, threshold = {threshold}")
    
    # Run gradient descent
    params_history, nll_history, grad_norm_history, iterations = gradient_descent(
        params_init, data, step_size, threshold
    )
    
    # Print results
    final_mu, final_sigma = params_history[-1]
    print("\nResults:")
    print(f"Total iterations: {iterations}")
    print(f"Final parameters: mu = {final_mu:.6f}, sigma = {final_sigma:.6f}")
    print(f"Final negative log-likelihood: {nll_history[-1]:.6f}")
    print(f"Final gradient norm: {grad_norm_history[-1]:.6f}")
    
    # Calculate analytical solution for comparison
    analytical_mu = np.mean(data)
    analytical_sigma = np.std(data, ddof=0)  # Using population standard deviation
    analytical_nll = negative_log_likelihood([analytical_mu, analytical_sigma], data)
    print("\nAnalytical solution:")
    print(f"mu = {analytical_mu:.6f}, sigma = {analytical_sigma:.6f}")
    print(f"Negative log-likelihood: {analytical_nll:.6f}")
    
    # Plot the negative log-likelihood vs. iterations
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(nll_history)), nll_history)
    plt.xlabel('Iterations')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('NLL vs. Iterations')
    plt.grid(True)
    
    # Plot the gradient norm vs. iterations
    plt.subplot(1, 2, 2)
    plt.plot(range(len(grad_norm_history)), grad_norm_history)
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm vs. Iterations')
    plt.grid(True)
    plt.tight_layout()
    
    # Plot the parameter trajectory
    plt.figure(figsize=(10, 8))
    
    # Create a meshgrid for contour plot
    mu_range = np.linspace(final_mu - 3, final_mu + 3, 100)
    sigma_range = np.linspace(max(0.1, final_sigma - 3), final_sigma + 3, 100)
    MU, SIGMA = np.meshgrid(mu_range, sigma_range)
    NLL = np.zeros_like(MU)
    
    # Calculate NLL values for contour plot
    for i in range(len(mu_range)):
        for j in range(len(sigma_range)):
            NLL[j, i] = negative_log_likelihood([MU[j, i], SIGMA[j, i]], data)
    
    # Plot contour and parameter trajectory
    contour = plt.contour(MU, SIGMA, NLL, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Negative Log-Likelihood')
    plt.plot(params_history[:, 0], params_history[:, 1], 'r.-', linewidth=1, markersize=3)
    plt.plot(params_history[0, 0], params_history[0, 1], 'go', markersize=10, label='Initial Parameters')
    plt.plot(params_history[-1, 0], params_history[-1, 1], 'mo', markersize=10, label='Final Parameters')
    plt.plot(analytical_mu, analytical_sigma, 'b*', markersize=10, label='Analytical Solution')
    plt.xlabel('μ (Mean)')
    plt.ylabel('σ (Standard Deviation)')
    plt.title('Parameter Trajectory in Gradient Descent')
    plt.legend()
    plt.grid(True)
    
    # Plot the data histogram with fitted normal distribution
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data Histogram')
    
    # Plot the fitted normal distribution
    x = np.linspace(min(data), max(data), 1000)
    fitted_pdf = 1/(final_sigma * np.sqrt(2*np.pi)) * np.exp(-(x - final_mu)**2 / (2 * final_sigma**2))
    plt.plot(x, fitted_pdf, 'r-', linewidth=2, label=f'Fitted Normal: μ={final_mu:.2f}, σ={final_sigma:.2f}')
    
    # Plot the analytical normal distribution
    analytical_pdf = 1/(analytical_sigma * np.sqrt(2*np.pi)) * np.exp(-(x - analytical_mu)**2 / (2 * analytical_sigma**2))
    plt.plot(x, analytical_pdf, 'b--', linewidth=2, label=f'Analytical: μ={analytical_mu:.2f}, σ={analytical_sigma:.2f}')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Data Histogram with Fitted Normal Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()
