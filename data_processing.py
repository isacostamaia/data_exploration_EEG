import math
import copy
import numpy as np
from scipy.linalg import eigh, pinv, norm
from scipy.signal import argrelextrema
import statistics
from moabb.paradigms import P300
import matplotlib.pyplot as plt


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


# for doing unit test:
# numerical_error = norm((B)@Cs@(B.T) - np.identity(Cs.shape[0])) #if filter is correct ~0
# numerical_error2 = norm(B@C_bar_class@(B.T) - np.diag(eigen_vals)) #if filter is correct ~0
# numerical_error, numerical_error, np.diag(B_p@C_bar_class@(B_p.T)) #should be the p highest eigenvals



def initialize_weights(class_epochs):
    weights = [1/np.linalg.norm(e_i) for e_i in class_epochs] #1/Frobenius_norm
    weights = weights/sum(weights) #normalize
    return weights

# apply weights in epochs before calculating filter

# def compute_weights(class_epochs, A_p, B_p):
#     """
#     Calculates set of weights for epochs from a given class
#         class_epochs: epochs from a given class
#         A_p,B_p: spatial filters
#     Returns: list of weights of len(class_epochs)
#     """
#     weights = [np.linalg.norm(A_p@B_p@e_i) - np.linalg.norm(A_p@B_p@e_i) for e_i in class_epochs]
#     weights = weights/sum(weights)
#     return weights

def improved_compute_weights(class_epochs, spatial_filter):
    """
    Calculates a set of weights for epochs from a given class
        class_epochs: epochs from a given class
        spatial_filter: fitted spatial filter
    Returns: list of weights of len(class_epochs)
    """
    class_epochs_sf = spatial_filter.apply(class_epochs)
    weights = [np.linalg.norm(e_sf_i) - np.linalg.norm(e_i - e_sf_i) for e_sf_i, e_i in zip(class_epochs_sf,class_epochs)]
    weights = weights/sum(weights)
    return weights

def find_local_max_idx(array, smooth= False, n_smooth=3):
    """
    Find indexes of local maxima values respecting that each local maxima is >= 66% of the global maxima
    Returns: indexes of local maxima
    """
    #smooth by averaging with n_neigh neighbour points
    if smooth:
        array = np.convolve(array, np.ones(n_smooth)/n_smooth, mode='same') 
    max_idx = argrelextrema(array, np.greater)[0]
    #get valid idx i.e. local max >= 66% of max
    valid_idx = np.where(array>=0.66*np.max(array))[0]
    #get valid local extrema idx
    valid_local_max_idx = valid_idx[np.isin(valid_idx, max_idx)]
    return valid_local_max_idx

def lagged_epochs(epoch, E):
    """
        For a given epoch, creates a list of lagged versions of it of len=2*E+1. Creates central sample [E:-E] and 2*E lagged versions around it
        Returns: list of lagged epochs
    """
    lagged_e_is=[]
    for eps in range(2*E+1):
        e_i = copy.deepcopy(epoch)
        e_i.crop(tmin=e_i.times[eps], tmax = e_i.times[len(e_i.times)+eps-2*E-1], include_tmax=True)
        lagged_e_is.append(e_i)
        del e_i
    return lagged_e_is

def apply_lags(epochs, E, lags_list):
    """
    Given a list of lags, and full uncropped epochs with its cropping parameter E, recreates cropped epochs lagged of the list value
    Returns
    """

    #save tmin from epoch that will be used as reference to rebuild epochs object
    ref_epoch_id = np.argmin(lags_list) 
    ref_epoch_tmin = epochs[ref_epoch_id].times[0] #tmn, info, events, metadata

    epochs_data = epochs.get_data()
    epochs_data_cropped_lagged = np.stack([e_i[E+l:-E+l] for e_i, l in zip(epochs_data, lags_list)])
    # epochs.__dict__.keys()
    # https://mne.tools/stable/generated/mne.EpochsArray.html

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
                                    drop_log = epochs.drop_log
                                    )
    return lagged_epochs
                        
def compute_lags(class_epochs, similarity="covariance", criteria_sim="local_max"):
    """
    Compute, for each epoch, the lag that amounts to the highest covariance between each epoch and the exclusive epochs average
    Returns: list of lags of len(class_epochs)
    """
    
    sfreq = class_epochs.info['sfreq'] #sampling frequency
    # E = math.floor(40*1e-3*sfreq) #Maximum allowed time-shift in samples unit. It should correspond to something around and less than 40ms 
    E = 3
    max_num_it = 2*E
    class_epochs_cropped = copy.deepcopy(class_epochs).crop(tmin=class_epochs.times[E], tmax = class_epochs.times[-E-1], include_tmax=True) #will only look to an epoch in window interval so we can use border values to compute the lag
    print("len class_epochs_cropped",len(class_epochs_cropped))
    for i, e_i in enumerate(class_epochs_cropped.iter_evoked()):
        print(e_i.get_data()[0].shape)
        print(e_i.get_data()[0])
        plt.figure()
        plt.scatter(e_i.times,e_i.get_data()[0])

    cond = E-1
    epochs_idx = np.arange(len(class_epochs_cropped))
    cond_hist = []
    num_it = 0


    while cond < E and num_it < max_num_it:
        lags_list=[]
        for i, e_i in enumerate (class_epochs.iter_evoked()):
            
            #(filtered and weighted) ensemble average excluding the current epoch/sweep
            avg_epochs_m1 = class_epochs_cropped[np.where(epochs_idx!= i)[0]].average(picks="all").get_data()
            #set of (filtered and weighted) single lagged i epoch estimation, for all lags
            lagged_e_is = lagged_epochs(e_i, E) #lags between -E and +E
            
            if similarity == "covariance":
                sim = np.array([(1/l_ei.get_data().shape[-1])*np.matrix.trace(l_ei.get_data()@avg_epochs_m1.T) for l_ei in lagged_e_is]).ravel()
            if similarity == "correlation":
                sim = np.array([np.corrcoef(l_ei.get_data()[0], avg_epochs_m1)[0,1] for l_ei in lagged_e_is])
                
            if criteria_sim == "local_max":
                best_idx = find_local_max_idx(sim)
                if best_idx.size>0:
                    best_idx = min(best_idx) #get smallest index (corresponds to the smallest lag)
                else:
                    best_idx = E #if there is no local max that matches constraints, the lag is 0
            if criteria_sim == "global_max":
                best_idx = np.argmax(sim) #if values are equal, by default takes the smallest argument
                
            best_idx = best_idx - E #reset reference. Sample [E:-E] corresponds to lag=0, sample [0:len(e_i)-2E] has lag =-E etc.
            lags_list.append(best_idx)          

        cond = sum(lags_list)
        cond_hist.append(cond)
        num_it+=1
        print(num_it)
        
    return lags_list, cond_hist
    