# Visualizing data
import matplotlib.pyplot as plt

def plot_features(X_train, Y_train, Y_hat):
    """
    Creates plots for individual features

    Args:
        X_train (ndarray): Contains independent variables
        Y_train (ndarray): Contains dependent variables
        Y_hat (ndarray): Untrained model predictions

    Returns:
        None

    """

    # Number of features
    n = X_train.shape[1]

    # Create figure and axis
    fig, axs = plt.subplots(nrows=n, figsize=(6, 4*n))

    # Iterate over features
    for j in range(n):

        # Plot the features
        axs[j].scatter(X_train[:, j], Y_train, label="Real")
        axs[j].scatter(X_train[:, j], Y_hat, color='r', label="Predicted")
        axs[j].set_title(f"Feature {j} vs. Target")
        axs[j].set_xlabel(f"Feature {j}")
        axs[j].set_ylabel("Target")
        axs[j].legend()

    # Show the plot
    plt.tight_layout()
    plt.show
