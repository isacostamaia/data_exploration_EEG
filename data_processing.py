import math
import copy
import numpy as np
from scipy.linalg import eigh, pinv
from scipy.signal import argrelextrema
import statistics
import mne
from moabb.paradigms import P300
import matplotlib.pyplot as plt


def get_clean_epochs(dataset, subjects_list=[1], paradigm = P300(), reject_value = 100e-6):
    """
    Get epochs from  dataset with epochs with values above reject_value excluded
    Returns: clean epochs
    """     
    # Fetch data for specific subjects
    epochs, _, _ = paradigm.get_data(dataset=dataset, subjects=subjects_list, return_epochs=True) #epochs, labels, metadata 
    print("Dataset median value: ", statistics.median(epochs.get_data().ravel()))
    
    #TODO: alter reject_value to be adaptive according to the dataset unit
    
    #reject epochs if a channel amplitude exceeds max value
    reject_criteria = dict(eeg=reject_value)  # 100 µV
    epochs.drop_bad(reject=reject_criteria)
     
    return epochs
    
class SpatialFilter:
    """
    Spatial filter based on epochs object and a given class_ 


    attributes: 
            epochs: epochs to apply the filter
            p: number of spatial filter components
            A_p: n_channels x p dimensional filter matrix
            B_p: p x n_channels dimensional filter matrix
    """
    def __init__(self, epochs, p = 4):
        
        self.p = p
        self.epochs = copy.deepcopy(epochs)
        
    def fit(self,class_ = "Target"):
        """
        Fit spatial filter in class_ of epochs object
            class_: class to which filter is fitted
    
        """
        epochs_data = self.epochs.get_data()*1e6 #n_epochs x n_channels x n_times using epochs from all classes
        covs = np.stack([e_i@e_i.T/e_i.shape[-1] for e_i in epochs_data]) #n_epochs x n_channels x n_channels
        Cs = np.mean(covs, axis=0) #n_channels x n_channels
        
        X_bar_class = self.epochs[class_].average().get_data()*1e6
        C_bar_class = (1/X_bar_class.shape[-1])* (X_bar_class)@(X_bar_class.T) #n_channels x n_channels
    
        eigen_vals, B = eigh(C_bar_class, Cs)
    
        A = pinv(B)
        self.A_p = A[-self.p:,:].T
        self.B_p = B[:,-self.p:].T
        

    def apply(self,epochs):
        """
        Returns filtered epochs (not inplace)
        """
        epoch_denoiser = lambda epoch: (self.A_p)@(self.B_p)@epoch
        epochs = copy.deepcopy(epochs)
        epochs.apply_function(epoch_denoiser, picks='all', channel_wise=False)
        return epochs


# TODO SpatialFilter unit tests:
# numerical_error = norm((B)@Cs@(B.T) - np.identity(Cs.shape[0])) #if filter is correct ~0
# numerical_error2 = norm(B@C_bar_class@(B.T) - np.diag(eigen_vals)) #if filter is correct ~0
# numerical_error, numerical_error, np.diag(B_p@C_bar_class@(B_p.T)) #should be the p highest eigenvals



def initialize_weights(class_epochs):
    weights = [1/np.linalg.norm(e_i) for e_i in class_epochs] #1/Frobenius_norm
    weights = weights/sum(weights) #normalize
    return weights


def compute_weights(class_epochs, spatial_filter):
    """
    Calculates a set of weights for epochs from a given class
        class_epochs: unfiltered epochs from a given class
        spatial_filter: fitted spatial filter
    Returns: list of weights of len(class_epochs)
    """
    class_epochs_sf = spatial_filter.apply(class_epochs)
    weights = [np.linalg.norm(e_sf_i) - np.linalg.norm(e_i - e_sf_i) for e_sf_i, e_i in zip(class_epochs_sf,class_epochs)]
    weights = weights/sum(weights)
    return weights

def apply_weights(epochs, weights):
    """
    Given epochs object and weights list, returns epochs object of the same shape, but with each epoch i scaled by the corresponding weight i
    epochs: epochs object of shape (n_epochs,n_channels,n_times)
    weights: list of length = n_epochs    
    """
    # wrapper function that uses the weights
    def create_epoch_multiplier_function(weights):
        epoch_counter = iter(weights)  # iterator over the weights
    
        def func(epoch_array):
            # next weight from the list
            multiplier = next(epoch_counter)
            return epoch_array * multiplier

        return func
        
    epochs_ = copy.deepcopy(epochs)

    epochs_.apply_function(create_epoch_multiplier_function(weights), picks="all", channel_wise=False)
    return epochs_

class Lagger:
    pass

def find_local_max_idx(array, valid_criteria=None, smooth=False, n_smooth=3):
    """
    Find indexes of local maxima value.
    If valid_criteria = "strict", local maximas should be >= 66% of the global maxima,
    Returns: indexes of local maxima
    """
    #smooth by averaging with n_neigh neighbour points
    if smooth:
        array = np.convolve(array, np.ones(n_smooth)/n_smooth, mode='same') 
    max_idx = argrelextrema(array, np.greater)[0]
    #get valid idx 
    if valid_criteria == "strict":
        #  local max >= 66% of max
        valid_idx = np.where(array>=0.66*np.max(array))[0]
        #get valid local extrema idx
        valid_idx = valid_idx[np.isin(valid_idx, max_idx)]
    else:
        valid_idx = max_idx
    return valid_idx

def lagged_epochs(epoch, E):
    """
        For a given epoch, create a list of lagged versions of it of len=2*E+1. Creates central sample [E:-E] and 2*E lagged versions around it
        Returns: list of lagged epochs
    """
    lagged_e_is=[]
    for eps in range(2*E+1):
        e_i = copy.deepcopy(epoch)
        e_i.crop(tmin=e_i.times[eps], tmax = e_i.times[len(e_i.times)+eps-2*E-1], include_tmax=True)
        lagged_e_is.append(e_i)
        del e_i  
    return lagged_e_is

def apply_lags(epochs, E, lags_list): #should be called correct_lags
    """
    Given a list of lags of length equal to the number of epochs in the epochs object, and full non-cropped epochs with its cropping parameter E, recreates cropped epochs lagged (corrected) of the list value
    epochs: full non-cropped epochs
    E: maximum lag
    lags_list: list of lags of length equal to number of epochs
    Returns
    """

    #save tmin from epoch that will be used as reference to rebuild epochs object
    ref_epoch_id = np.argmin(lags_list) 
    ref_epoch_tmin = epochs[ref_epoch_id].times[E]  #get time of ref_epoch at cropping sample as tmin


    epochs_data = epochs.get_data() 
    epochs_data_cropped_lagged = np.stack([e_i[:,E+l:len(epochs.times)-E+l] for e_i, l in zip(epochs_data, lags_list)])

    lagged_epochs = mne.EpochsArray(epochs_data_cropped_lagged,
                                        info = epochs.info, 
                                        events = epochs.events,
                                        tmin= ref_epoch_tmin,
                                        event_id = epochs.event_id, 
                                        reject = epochs.reject,
                                        baseline = epochs.baseline,
                                        proj = epochs.proj,
                                        metadata = epochs.metadata,
                                        selection = epochs.selection,
                                        drop_log = epochs.drop_log, 
                                        verbose = 'WARNING'
                                        )
    return lagged_epochs
                        
def compute_lags(class_epochs, similarity="covariance", criteria_sim="greatest_local_max", E=None):
    """
    Compute, for each epoch, the lag that amounts to the highest covariance between each epoch and the exclusive epochs average
    E: Maximum allowed time-shift in samples unit
    Returns: list of lags of len(class_epochs)
    """
    if not E:
        sfreq = class_epochs.info['sfreq'] #sampling frequency
        E = math.floor(40*1e-3*sfreq) #Maximum allowed time-shift in samples unit. It should correspond to something around and less than 40ms 
    print("E: ", E)
    max_num_it = 2*E
    class_epochs_cropped = copy.deepcopy(class_epochs).crop(tmin=class_epochs.times[E], tmax = class_epochs.times[-E-1], include_tmax=True) #will only look to an epoch in window interval so we can use border values to compute the lag

    cond = E-1
    epochs_idx = np.arange(len(class_epochs_cropped))
    cond_hist = []
    num_it = 0


    while cond < E and num_it < max_num_it:
        lags_list=[]
        print("Iteration num: ", num_it)
        for i, e_i in enumerate (class_epochs.iter_evoked()):
            #(filtered and weighted) ensemble average excluding the current epoch/sweep
            avg_epochs_m1 = class_epochs_cropped[np.where(epochs_idx!= i)[0]].average(picks="all").get_data()
            #set of (filtered and weighted) single lagged i epoch estimation, for all lags
            lagged_e_is = lagged_epochs(e_i, E) #lags between -E and +E
            
            if similarity == "covariance":
                sim = np.array([(1/l_ei.get_data().shape[-1])*np.matrix.trace(l_ei.get_data()@avg_epochs_m1.T) for l_ei in lagged_e_is]).ravel()
            elif similarity == "correlation":
                sim = np.array([np.corrcoef(l_ei.get_data()[0], avg_epochs_m1)[0,1] for l_ei in lagged_e_is])
            else:
                raise ValueError("similarity should be amongst 'covariance' and 'correlation'")
                
            if criteria_sim == "strict_local_max": #get local max that is greater or equal to 66% of global max and minimizes the lag
                best_idx = find_local_max_idx(sim, valid_criteria="strict") - E #reset reference. Sample [E:-E] corresponds to lag=0, sample [0:len(e_i)-2E] has lag =-E etc.
                if best_idx.size>0:
                    best_idx = min(best_idx, key=abs) #get the smallest index (corresponds to the smallest lag)
                else:
                    best_idx = 0 #if there is no local max that matches constraints, the lag is 0
         
            elif criteria_sim == "local_max_min_lag": #get local max that minimizes the lag
                best_idx = find_local_max_idx(sim, valid_criteria=None) - E #reset reference. Sample [E:-E] corresponds to lag=0, sample [0:len(e_i)-2E] has lag =-E etc.
                if best_idx.size>0:
                    best_idx = min(best_idx, key=abs) #get the smallest index (corresponds to the smallest lag)
                else:
                    best_idx = 0 #if there is no local max that matches constraints, the lag is 0            
            
            elif criteria_sim == "greatest_local_max": #get greatest local maxima (regardless of min lag)
                best_idx = find_local_max_idx(sim, valid_criteria=None)
                if best_idx.size>0:
                    best_idx = best_idx[np.argmax(sim[best_idx])] #get index of greatest local max
                    best_idx = best_idx - E #reset reference. Sample [E:-E] corresponds to lag=0, sample [0:len(e_i)-2E] has lag =-E etc.
                else:
                    best_idx = 0 #if there is no local max that matches constraints, the lag is 0
           
            elif criteria_sim == "global_max": #get simply the local max, regardless of lag value
                best_idx = np.argmax(sim) - E #reset reference. Obs. if values in sim are equal, by default takes the smallest argument
            else:
                raise ValueError("criteria_sim should be amongst 'strict_local_max', 'local_max_min_lag', \
                'greatest_local_max', 'global_max' ")
            lags_list.append(best_idx) 

        #update class_epochs_cropped
        class_epochs_cropped =  apply_lags(class_epochs, E, lags_list)       
        cond = sum(abs(lag) for lag in lags_list)
        cond_hist.append(cond)
        num_it+=1
        
    return lags_list, cond_hist
    