import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from io import StringIO
import math

# Function to download and prepare the dataset
def load_data():
    url = "https://docs.google.com/spreadsheets/d/13CmIStaYtiQqR_dhBPrkHJINvVln9cepHypNinVQT3c/export?format=csv&gid=2023320122"
    
    try:
        # Try to download the data
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the CSV data
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        # Extract the data
        data = df.values.flatten()  # Convert to 1D array
        
        return data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create some dummy data if loading fails
        print("Generating dummy data instead...")
        np.random.seed(42)
        data = np.random.normal(5, 2, 100)  # Generate normally distributed data
        return data

# Negative log-likelihood function for normal distribution
def cost_function(params, data):
    mu, sigma = params
    n = len(data)
    
    # Ensure sigma is positive
    if sigma <= 0:
        return float('inf')
    
    # Compute negative log-likelihood
    log_likelihood = n * np.log(sigma) + np.sum((data - mu)**2) / (2 * sigma**2)
    
    return log_likelihood

# Gradient of the negative log-likelihood function
def gradient(params, data):
    mu, sigma = params
    n = len(data)
    
    # Ensure sigma is positive
    if sigma <= 0:
        return np.array([0, 0])  # Return zero gradient if sigma is invalid
    
    # Compute gradients
    grad_mu = -np.sum(data - mu) / (sigma**2)
    grad_sigma = n / sigma - np.sum((data - mu)**2) / (sigma**3)
    
    return np.array([grad_mu, grad_sigma])

# Gradient descent implementation
def gradient_descent(data, params_init, eta, epsilon, max_iterations=10000):
    params = params_init.copy()
    cost_history = []
    params_history = [params.copy()]
    
    for i in range(max_iterations):
        # Calculate current cost
        current_cost = cost_function(params, data)
        cost_history.append(current_cost)
        
        # Calculate gradient
        grad = gradient(params, data)
        grad_norm = np.linalg.norm(grad)
        
        # Check convergence
        if grad_norm < epsilon:
            print(f"Converged after {i+1} iterations. Gradient norm: {grad_norm:.8f}")
            break
        
        # Update parameters
        params = params - eta * grad
        
        # Ensure sigma remains positive
        params[1] = max(params[1], 1e-10)
        
        params_history.append(params.copy())
        
        # Optional: print progress every 100 iterations
        if (i+1) % 100 == 0:
            print(f"Iteration {i+1}: Cost = {current_cost:.6f}, Gradient norm = {grad_norm:.6f}")
            print(f"Current parameters: mu = {params[0]:.6f}, sigma = {params[1]:.6f}")
    
    if i == max_iterations - 1:
        print(f"Maximum iterations ({max_iterations}) reached. Gradient norm: {grad_norm:.8f}")
    
    return params, cost_history, np.array(params_history)

# Function to plot results
def plot_results(cost_history, params_history, data):
    # Plot 1: Cost vs iterations
    plt.figure(figsize=(15, 5))
    
    # Plot cost history
    plt.subplot(1, 3, 1)
    plt.plot(cost_history)
    plt.title('Cost vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Negative Log-Likelihood')
    plt.grid(True)
    
    # Plot 2: Parameters path
    plt.subplot(1, 3, 2)
    plt.plot(params_history[:, 0], params_history[:, 1], 'g-', alpha=0.5)
    plt.plot(params_history[0, 0], params_history[0, 1], 'go', label='Start')
    plt.plot(params_history[-1, 0], params_history[-1, 1], 'r*', label='End')
    plt.title('Optimization Path')
    plt.xlabel('μ')
    plt.ylabel('σ')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Data histogram with fitted normal distribution
    plt.subplot(1, 3, 3)
    plt.hist(data, bins=30, density=True, alpha=0.6, label='Data')
    
    # Get final parameters
    mu_final, sigma_final = params_history[-1]
    
    # Plot fitted normal distribution
    x = np.linspace(min(data), max(data), 1000)
    y = (1 / (sigma_final * np.sqrt(2 * np.pi))) * np.exp(-(x - mu_final)**2 / (2 * sigma_final**2))
    plt.plot(x, y, 'r-', linewidth=2, label=f'Fitted N({mu_final:.2f}, {sigma_final:.2f})')
    
    plt.title('Data and Fitted Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    print("Loading data...")
    data = load_data()
    
    # Problem parameters
    params_init = np.array([0, 1])  # Initial point: mu_0 = 0, sigma_0 = 1
    eta = 0.01                      # Step size
    epsilon = 1e-5                  # Convergence threshold
    
    print(f"Starting gradient descent with:")
    print(f"- Initial parameters: mu = {params_init[0]}, sigma = {params_init[1]}")
    print(f"- Step size (η): {eta}")
    print(f"- Convergence threshold (ε): {epsilon}")
    
    # Run gradient descent
    params_final, cost_history, params_history = gradient_descent(data, params_init, eta, epsilon)
    
    # Display results
    print("\nFinal solution:")
    print(f"μ = {params_final[0]:.6f}")
    print(f"σ = {params_final[1]:.6f}")
    print(f"Final negative log-likelihood: {cost_history[-1]:.6f}")
    print(f"Number of iterations: {len(cost_history)}")
    
    # Calculate sample statistics for comparison
    print("\nSample statistics:")
    print(f"Sample mean: {np.mean(data):.6f}")
    print(f"Sample standard deviation: {np.std(data, ddof=1):.6f}")
    
    # Plot results
    plot_results(cost_history, params_history, data)

if __name__ == "__main__":
    main()
