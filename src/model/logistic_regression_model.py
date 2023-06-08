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

    return Y_hat

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

    # Print diagnostics
    print(f"""
           ########## INITIAL PREDICTIONS ##########
           
           {predictions}""")
