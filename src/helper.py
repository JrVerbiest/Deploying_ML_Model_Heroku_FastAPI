"""helper file
"""
import pickle

def load_pkl(pkl_path):

    pkl_file = open(pkl_path, "rb")
    object_file = pickle.load(pkl_file)
    pkl_file.close()

    return object_file


def save_pkl(data, pkl_path):
    """ save to a pkl-file.
    
    Parameters
    ----------
    data: 
        .
    pkl_path : str
        path
    
    Returns:
    --------
    None
      
    """
    
    with open(pkl_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
    pkl_file.close()