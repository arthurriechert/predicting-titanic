# Linear algebra
import numpy as np

# Data visualization
import matplotlib.pyplot as plt

# Getting data
import model.load_data as ld

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
