import numpy as np

def field_root_mean_square(class_epochs):
    
    class_epochs_data = class_epochs.get_data() #array of shape (n_epochs, n_channels, n_times)

    n = class_epochs_data.shape[1] #n_channels
    centering_matrix = np.eye(n) - np.ones((n, n)) / n
    #centered_Xk has each column with zero mean
    centered_Xks = [centering_matrix@Xk for Xk in class_epochs_data]

    #frms_i is the std of the columns of centered_Xk
    frms = np.stack([Xk.std(axis=0) for Xk in centered_Xks])  #each pfmrs_i has dim = n_times, frms will be n_epochs x n_times

    return frms