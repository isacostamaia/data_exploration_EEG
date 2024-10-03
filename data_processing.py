import copy
import numpy as np
from scipy.linalg import eigh, pinv, norm
import statistics
from moabb.paradigms import P300


def get_clean_epochs(dataset, subjects_list=[1], paradigm = P300(), reject_value = 100e-6):
    """
    Get epochs from  dataset with epochs with values above reject_value excluded
    Returns: clean epochs
    """     
    # Fetch data for specific subjects
    # You can specify subjects here; for now, we use two subjects as an example
    epochs, _, _ = paradigm.get_data(dataset=dataset, subjects=subjects_list, return_epochs=True) #epochs, labels, metadata 
    print("Dataset median value: ", statistics.median(epochs.get_data().ravel()))
    
    #TODO: alter reject_value to be adaptive according to the dataset unit
    
    #reject epochs if a channel amplitude exceeds max value
    reject_criteria = dict(eeg=reject_value)  # 100 µV
    epochs.drop_bad(reject=reject_criteria)
     
    return epochs
    
class SpatialFilter:

    def __init__(self, epochs, p = 4):
        
        self.p = p
        self.epochs = epochs
        
    def fit(self,class_ = "Target"):
        """
        Fit spatial filter based on epochs object and a given class_ 
            epochs: epochs to apply the filter
            p: number of components
            class_: class to which filter is fitted
    
        Returns: p-components filter to filter epochs (resultant is in sensor space)
        """
        epochs_data = self.epochs.get_data()*1e6 #n_epochs x n_channels x n_times using epochs from all classes
        covs = np.stack([e_i@e_i.T/e_i.shape[-1] for e_i in epochs_data]) #n_epochs x n_channels x n_channels
        Cs = np.mean(covs, axis=0) #n_channels x n_channels
        
        X_bar_class = self.epochs[class_].average().get_data()*1e6
        C_bar_class = (1/X_bar_class.shape[-1])* (X_bar_class)@(X_bar_class.T) #n_channels x n_channels
    
        eigen_vals, B = eigh(C_bar_class, Cs)
    
        A = pinv(B)
        self.A_p = A[-self.p:,:]
        self.B_p = B[:,-self.p:]
        

    def apply(self,epochs):
        epoch_denoiser = lambda epoch: (self.A_p.T)@(self.B_p.T)@epoch
        epochs = copy.deepcopy(epochs)
        epochs.apply_function(epoch_denoiser, picks='all', channel_wise=False)
        return epochs




    