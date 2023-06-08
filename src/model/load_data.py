# Loading data
import pandas as pd

# Linear algebra
import numpy as np

def correct_nan_values(np_array):
    """
    Returns an np_array with Nan values replaced for the mean

    Args:
        np_array (ndim): An n-dimensional numpy array

    Returns:
        np_array (ndim): An n-dimensional numpy array

    """
    
    # Check the dimensions
    dimension = np_array.ndim

    # Placeholder
    correct_array = np.empty(np_array.shape)

    # Determine if it is 1-D or 2-D
    if dimension == 1:
        
        # Get the mean, ignoring nan values
        mean = np.nanmean(np_array)

        # Replace nan with the mean
        np_array =np.where(np.isnan(np_array), mean, np_array)
    
    elif dimension == 2:

        # Loop through each column
        for i in range(np_array.shape[1]):

            # Get the mean for each column, ignoring nan values
            mean = np.nanmean(np_array[:, i])

            # Replace nan with the mean of the respective column
            np_array[:, i] = np.where(np.isnan(np_array[:, i]), mean, np_array[:, i])

    return np_array

def get_training_sets():
    """
    Loads training sets from csv into numpy array

    Args:
        None

    Returns:
        X_train (ndarray): Numpy array, shape (m, n), where m is number of training examples and n is number of features
        Y_train (ndarray): Numpy array, shape (,m), where m is number of training examles

    """

    # Load data from csv
    data = pd.read_csv("../data/train.csv")

    # Organize features, refer to README for more info
    data["Age x Fare"] = data["Age"] * data["Fare"]
    X_train = data[["Age", "Fare", "Age x Fare"]].to_numpy()

    # Remove nan
    X_train = correct_nan_values(X_train)

    # Organize dependent variables
    Y_train = data["Survived"]

    # Remove nan
    Y_train = correct_nan_values(Y_train)
    
    # Print diagnostics
    print(f"""
           ######### X TRAIN  ######### 
        
           \033[32m{X_train}\033[0m


           ######### Y TRAIN  ######### 
          
           \033[34m{Y_train}\033[0m


            ######### DETAILS #########
           
           \033[32mShape of X: {X_train.shape}\033[0m

           \033[34mShape of Y: {Y_train.shape}\033[0m

           \033[32mData Type of X: {X_train.dtype}\033[0m

           \033[34mData Type of Y: {Y_train.dtype}\033[0m

           \033[32mDimensions of X: {X_train.ndim}\033[0m

           \033[34mDimensions of Y: {Y_train.ndim}\033[0m

           \033[32mX NaN Values: {np.any(np.isnan(X_train))}\033[0m

           \033[34mY NaN Values: {np.any(np.isnan(Y_train))}\033[0m""")

    return X_train, Y_train

def get_testing_sets():
    """
    Loads testing sets from csv into numpy array

    Args:
        None

    Returns:
        X_test (ndarray): Numpy array, shape (m, n), where m is number of testing examples and n is number of features
        Y_test (ndarray): Numpy array, shape (,m), where m is number of testing examples

    """

    # Load data from csv
    data = pd.read_csv("../data/train.csv")
    
    # Organize features, refer to README for more info
    data["Age x Fare"] = data["Age"] * data["Fare"]
    X_test = data[["Age", "Fare", "Age x Fare"]].to_numpy()
   
    # Remove nan
    X_test = correct_nan_values(X_test)

    # Organize dependent variables
    Y_test = data["Survived"]

    # Remove nan
    Y_test = correct_nan_values(Y_test)
    
    # Print diagnostics
    print(f"""
           ######### X TEST  ######### 
        
           \033[36m{X_test}\033[0m


           ######### Y TEST  ######### 
          
           \033[37m{Y_test}\033[0m


            ######### DETAILS #########
           
           \033[36mShape of X: {X_test.shape}\033[0m

           \033[37mShape of Y: {Y_test.shape}\033[0m

           \033[36mData Type of X: {X_test.dtype}\033[0m

           \033[37mData Type of Y: {Y_test.dtype}\033[0m

           \033[36mDimensions of X: {X_test.ndim}\033[0m

           \033[37mDimensions of Y: {Y_test.ndim}\033[0m
          
           \033[32mX NaN Values: {np.any(np.isnan(X_test))}\033[0m

           \033[34mY NaN Values: {np.any(np.isnan(Y_test))}\033[0m""")

    return X_test, Y_test
