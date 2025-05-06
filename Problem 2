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
        
        # Extract features (X) and labels (y)
        # Assuming the last column is the target variable and the rest are features
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # Add bias term
        X_with_bias = np.column_stack((np.ones(X.shape[0]), X))
        
        return X_with_bias, y
    except Exception as e:
        print(f"Error loading data: {e}")
        print("If you haven't downloaded the data yet, please download it from the Google Sheets link as a CSV file.")
        return None, None

# Sigmoid function
def sigmoid(z):
    """Compute the sigmoid function"""
    return 1 / (1 + np.exp(-z))

# Define the negative log-likelihood function for logistic regression
def negative_log_likelihood(beta, X, y):
    """
    Calculate negative log-likelihood for logistic regression
    
    Args:
        beta: Parameter vector
        X: Feature matrix with bias term
        y: Target labels (0 or 1)
        
    Returns:
        Negative log-likelihood value
    """
    z = X @ beta
    nll = np.sum(np.log(1 + np.exp(-y * z)))
    return nll

# Define the gradient of the negative log-likelihood
def gradient(beta, X, y):
    """
    Calculate gradient of negative log-likelihood
    
    Args:
        beta: Parameter vector
        X: Feature matrix with bias term
        y: Target labels (0 or 1)
        
    Returns:
        Gradient vector
    """
    z = X @ beta
    # Calculate the gradient
    return -np.sum(y[:, np.newaxis] * X * (1 - sigmoid(y * z))[:, np.newaxis], axis=0)

# Implement gradient descent
def gradient_descent(beta_init, X, y, step_size, threshold, max_iterations=10000):
    """
    Perform gradient descent to minimize negative log-likelihood
    
    Args:
        beta_init: Initial parameters
        X: Feature matrix with bias term
        y: Target labels (0 or 1)
        step_size: Learning rate
        threshold: Convergence threshold for gradient norm
        max_iterations: Maximum number of iterations
        
    Returns:
        beta_history: History of parameters
        nll_history: History of negative log-likelihood values
        grad_norm_history: History of gradient norms
        iterations: Number of iterations performed
    """
    beta = beta_init.copy()
    iterations = 0
    beta_history = [beta.copy()]
    nll_history = [negative_log_likelihood(beta, X, y)]
    grad_norm_history = []
    
    while iterations < max_iterations:
        grad = gradient(beta, X, y)
        grad_norm = np.linalg.norm(grad)
        grad_norm_history.append(grad_norm)
        
        if grad_norm < threshold:
            print(f"Converged after {iterations} iterations!")
            break
            
        # Update parameters
        beta = beta - step_size * grad
        
        beta_history.append(beta.copy())
        nll_history.append(negative_log_likelihood(beta, X, y))
        iterations += 1
        
        # Print progress every 500 iterations
        if iterations % 500 == 0:
            print(f"Iteration {iterations}, NLL: {nll_history[-1]:.4f}, Gradient norm: {grad_norm:.6f}")
    
    return np.array(beta_history), np.array(nll_history), np.array(grad_norm_history), iterations

# Function to make predictions
def predict(beta, X):
    """Make predictions using the logistic regression model"""
    probabilities = sigmoid(X @ beta)
    return (probabilities >= 0.5).astype(int)

# Function to calculate accuracy
def accuracy(y_true, y_pred):
    """Calculate accuracy of predictions"""
    return np.mean(y_true == y_pred)

def main():
    # Load data - replace with your actual file path
    filepath = "logistic_data.csv"  # You'll need to download the Google Sheet as CSV
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"File {filepath} not found!")
        print("Please download the data from the Google Sheets link and save it as 'logistic_data.csv' in the same directory as this script.")
        # For demonstration, generate some random data
        print("Generating random data for demonstration...")
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int) * 2 - 1  # Binary labels {-1, 1}
        X_with_bias = np.column_stack((np.ones(X.shape[0]), X))
    else:
        X_with_bias, y = load_data(filepath)
        
    if X_with_bias is None or y is None:
        print("Using randomly generated data for demonstration...")
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int) * 2 - 1  # Binary labels {-1, 1}
        X_with_bias = np.column_stack((np.ones(X.shape[0]), X))
    
    # Parameters for gradient descent
    n_features = X_with_bias.shape[1]
    beta_init = np.zeros(n_features)  # Initial beta = [0, 0, ...]
    step_size = 0.05
    threshold = 1e-5
    
    print(f"Data shape: X = {X_with_bias.shape}, y = {y.shape}")
    print(f"Initial parameters: beta = {beta_init}")
    print(f"Starting gradient descent with step size = {step_size}, threshold = {threshold}")
    
    # Run gradient descent
    beta_history, nll_history, grad_norm_history, iterations = gradient_descent(
        beta_init, X_with_bias, y, step_size, threshold
    )
    
    # Print results
    final_beta = beta_history[-1]
    print("\nResults:")
    print(f"Total iterations: {iterations}")
    print(f"Final parameters: beta = {final_beta}")
    print(f"Final negative log-likelihood: {nll_history[-1]:.6f}")
    print(f"Final gradient norm: {grad_norm_history[-1]:.6f}")
    
    # Make predictions and calculate accuracy
    y_pred = predict(final_beta, X_with_bias)
    acc = accuracy(y, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
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
    
    # If we have 2D features, visualize the decision boundary
    if X_with_bias.shape[1] == 3:  # 2 features + bias
        plt.figure(figsize=(10, 8))
        
        # Extract original features without bias
        X_orig = X_with_bias[:, 1:]
        
        # Create a meshgrid for visualization
        x_min, x_max = X_orig[:, 0].min() - 1, X_orig[:, 0].max() + 1
        y_min, y_max = X_orig[:, 1].min() - 1, X_orig[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # Create feature matrix for prediction
        X_grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
        
        # Predict probabilities
        Z = sigmoid(X_grid @ final_beta)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
        plt.contour(xx, yy, Z, [0.5], linewidths=2, colors='black')
        
        # Plot data points
        for i, label in enumerate(np.unique(y)):
            mask = y == label
            plt.scatter(X_orig[mask, 0], X_orig[mask, 1], 
                       label=f'Class {label}', 
                       s=60, alpha=0.7,
                       edgecolors='k')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.legend()
        plt.grid(True)
        
        # Plot the path of beta values (for b1 and b2 only)
        plt.figure(figsize=(8, 6))
        plt.plot(beta_history[:, 1], beta_history[:, 2], 'r.-', linewidth=1, markersize=3)
        plt.plot(beta_history[0, 1], beta_history[0, 2], 'go', markersize=10, label='Initial')
        plt.plot(beta_history[-1, 1], beta_history[-1, 2], 'mo', markersize=10, label='Final')
        plt.xlabel('β₁')
        plt.ylabel('β₂')
        plt.title('Parameter Path during Gradient Descent')
        plt.legend()
        plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()
