import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne

def field_root_mean_square(class_epochs):
    """
    Calculates Field Root Mean Square
        Parameters:
            class_epochs: object corresponding to a class, could be EpochsArraay or ndarray
        Returns:
            df: instance of dataframe with row sorted field root mean square
    """
    print(type(class_epochs))
    
    if isinstance(class_epochs,mne.EpochsArray) or isinstance(class_epochs, mne.Epochs):
        class_epochs_data = class_epochs.get_data() #array of shape (n_epochs, n_channels, n_times)
    elif isinstance(class_epochs, np.ndarray):
        class_epochs_data = class_epochs  #array of shape (n_epochs, n_channels, n_times)
    else:
        raise TypeError(f"class_epochs must be EpochsArray or ndarray type, not {type(class_epochs)}")
    
    # n = class_epochs_data.shape[1] #n_channels
    # centering_matrix = np.eye(n) - np.ones((n, n)) / n
    # centered_Xks = [centering_matrix@Xk for Xk in class_epochs_data] #centered_Xk has each column with zero mean
    centered_Xks = class_epochs_data
    
    #frms_i is the std of the columns of centered_Xk
    frms = np.stack([Xk.std(axis=0) for Xk in centered_Xks])  #each pfmrs_i has dim = n_times, frms will be n_epochs x n_times

    #create df and sort by sum of the rows
    df = pd.DataFrame(frms, index = np.arange(1, frms.shape[0]+1), columns = ['%.f' % elem for elem in class_epochs.times*1e3])
    df["row_sum"]=df.apply(lambda x: sum(x), axis=1)
    df = df.sort_values("row_sum", ascending=False).drop(columns=['row_sum'])

    return df


def plot_fmrs(frms):

    fig, ((ax1, cbar_ax, ax3), (ax2, dummy_ax1, dummy_ax2)) = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), sharex='col',
                                                      gridspec_kw={'height_ratios': [5, 1], 'width_ratios': [30, 1, 8]})

    #FRMS
    g1 = sns.heatmap(frms, ax=ax1, cbar_ax=cbar_ax)
    ax1.set_title("FRMS")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Epoch")
    
    #Row sum
    ax3.plot(np.arange(0,len(frms)),frms.iloc[::-1].apply(lambda x: sum(x), axis=1))
    ax3.set_title("Row sum")
    ax3.yaxis.tick_right()
    
    #Trimmed mean of columns taking out 10% oh highest and lowest values
    ax2.plot(stats.trim_mean(frms, proportiontocut=0.1, axis=0))
    # ax2.set_title("Column trimmed mean")
    ax2.tick_params(which='major', labelrotation=90)
    
    
    dummy_ax1.set_axis_off()
    dummy_ax2.set_axis_off()

    plt.show()