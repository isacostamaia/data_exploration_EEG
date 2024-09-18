def field_root_mean_square(class_epochs):
    
    class_epochs_data = class_epochs.get_data() #array of shape (n_epochs, n_channels, n_times)
    Xk = np.reshape(class_epochs_data, (class_epochs_data.shape[1], class_epochs_data.shape[0]*class_epochs_data.shape[2])) #Xk in R^(NxT)

    n = Xk.shape[0]
    centering_matrix = np.eye(n) - np.ones((n, n)) / n
    #centered_Xk has each column with zero mean
    centered_Xk = centering_matrix@Xk

    #TODO:
    #phi is the std of the columns of centered_Xk
    pass