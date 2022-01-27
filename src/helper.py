"""helper file
"""
import pickle

def save_pkl(data, pkl_path):
    """ save trained model to a pkl-file.
    
    Parameters
    ----------
    data: 
        .
    pkl_path : str
        path to save the model.
    
    Returns:
    --------
    None
      
    """
    
    with open(pkl_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
    pkl_file.close()