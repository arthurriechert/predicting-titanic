# Linear algebra
import numpy as np

# Getting data
import model.load_data as ld

# Generating random parameters
import random

# Saving and loading parameters
import os

def random_parameter_init(n):
    """
    Produces a random set of parameters

    Args:
        n (int): Number of features

    Returns:
        W (ndarray): Random weights
        b (float): Scalar bias

    """

    # Add noise
    W = 0.02 * np.random.rand(n)
    
    # Randomize b
    b = 0.02 * round(random.uniform(1.0, 50.0), 3)

    return W, b


def sigmoid(z):
    """
    Applies sigmoid to f_wb

    Args:
        z (ndarray): A set of predictions

    Returns:
        g_z (ndarray): Predictions with sigmoid applied

    """

    z = np.clip(z, -500, 500)  # clip to avoid overflow in np.exp 

    # Apply sigmoid
    g_z = 1 / (1 + np.exp(-z))
       
    return g_z

def compute_predictions(W, X, b):
    """
    Computes prediction for a batch

    Args:
        W (ndarray): Contains weights of model
        X (ndarray): Contains independent variables
        b (float): Scalar bias

    Returns:
        Y_hat (ndarray): Contains predictions

    """

    # Compute the dot product
    Y_hat = np.dot(X,W) + b

    # Apply sigmoid
    Y_hat = sigmoid(Y_hat)

    return Y_hat

def compute_cost(W, X, b, Y):
    """
    Use log loss to compute the cost

    Args:
        W (ndarray): Contains weights for model
        X (ndarray): Contains independent variables
        b (float): Scalar bias
        Y (ndarray): Given answers

    Returns:
        cost (float): Scalar

    """

    # Get number of examples
    m = X.shape[0]

    # Get predictions
    Y_hat = compute_predictions(W, X, b)

    # Small constant to prevent log(0)
    eps = 1e-15

    # Clipping values to ensure they fall within [eps, 1-eps]
    Y_hat = np.clip(Y_hat, eps, 1 - eps)

    # Compute the loss
    loss = -Y * np.log(Y_hat) - (1 - Y) * np.log(1 - Y_hat)

    # Get the mean
    cost = np.mean(loss)

    return cost

def compute_gradient(W, X, b, Y):
    """
    Compute the partial derivatives

    Args:
        W (ndarray): Contains weights for models
        X (ndarray): Contains independent variables
        b (float): Scalar bias
        Y (ndarray): Given answers

    Returns:
        gradient_W (ndarray): Partial derivatives of cost with respect to W
        gradient_b (ndarray): Partial derivatives of cost with respect to b

    """

    # Get predictions
    Y_hat = compute_predictions(W, X, b)

    # Compute loss
    loss = Y_hat - Y

    # Compute gradients
    grad_W = np.dot(X.T, loss) / X.shape[0]
    grad_b = np.sum(loss) / X.shape[0]
    
    return grad_W, grad_b

def gradient_descent(W, X, b, Y, epochs, alpha):
    """
    Performs standard gradient descent

    Args:
        W (ndarray): Contains weights for models
        X (ndarray): Contains independent variables
        b (float) Scalar bias
        Y (ndarray): Given answers
        epochs (int): Number of times to perform gradient descent
        alpha (float): Learning rate

    Returns:
        W (ndarray): Update version of W
        b (float): Updated version of b

    """


    # Perform number of gradient descent
    for epoch in range(epochs):

        # Get the gradients
        grad_W, grad_b = compute_gradient(W, X, b, Y)

        # Get the cost
        cost = compute_cost(W, X, b, Y)

        # Update parameters
        W = W - alpha * grad_W
        b = b - alpha * grad_b

        # Print diagnostics every 100 iterations
        if epoch % 100 == 0:
            print(f"\n#{epoch} -> COST: {cost} | W: {W} | b: {b} | d_W: {grad_W} | d_b: {grad_b}")

    # Print diagnostic
    print(f"\n\033[32mW optimized at {W} | b optimized at {b}")

    return W, b

def run():
    """
    A basic setup to run the model

    Args:
        None

    Returns:
        None

    """

    # Load data
    X_train, Y_train = ld.get_training_sets()
    X_test, Y_test = ld.get_testing_sets()

    # Number of examples, features
    m, n = X_train.shape

    # Initialize parametesr
    W = np.empty((m, n))
    b = 0

    # Check if weights and biases already exist
    if not os.path.exists("model/weights.npy") or not os.path.exists("model/biases.npy"):

        # Print diagnostics
        print("\n\033[31mCouldn't find weights and biases. Initializing random parameters\033[0m")

        # Get random vaues
        W, b = random_parameter_init(n)

    else:
        
        # Print diagnostics
        print("\n\033[32mFound parameters, loading\033[0m")

        # Load from npy file
        W = np.load("model/weights.npy")
        b = np.load("model/biases.npy")


    # Print diagnostics
    print(f"\nWEIGHTS: {W}\nBIASES: {b}")

    # Compute initial predictions
    predictions = compute_predictions(W, X_train, b)

    # Apply sigmoid
    predictions = sigmoid(predictions)

    # Compute initial cost
    cost = compute_cost(W, X_train, b, Y_train)

    # Print diagnostics
    print(f"""
           ########## INITIAL PREDICTIONS ##########
           
           {predictions}


           ############# INITIAL COST ##############
            
           {cost}""")

    # Get settings
    alpha = float(input("\nLEARNING RATE > "))
    epochs = int(input("\nITERATIONS > "))

    # Create placeholders
    W_final = np.empty((m,n))
    b_final = 0

    W_final, b_final = gradient_descent(W, X_train, b, Y_train, epochs, alpha)
     
