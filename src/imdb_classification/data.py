import os 
import pandas as pd 

def load_imdb_data(data_dir, subset = 'train'):
    """
    Load IMDb review dataset. 

    Parameters:
    data_dir (str): directory containing the dataset. 
    subset (str): 'train' for train dataset, 'test' for test dataset.

    Returns:
    pd.DataFrame: column 'review' is the review (str), column 'label' is 0 for 
        a negative review, 1 for a positive review.
    """
    directory = os.path.join(data_dir, subset)
    neg_dir = os.path.join(directory, 'neg') 
    pos_dir = os.path.join(directory, 'pos')
    data = []
    for d, label in [(neg_dir, 0), (pos_dir, 1)]:
        for filename in os.listdir(d):
            path = os.path.join(d, filename) 
            with open(path, 'r', encoding = 'utf-8') as file:
                review = file.read() 
            data.append({'label': label, 'review': review})   
    return pd.DataFrame(data)