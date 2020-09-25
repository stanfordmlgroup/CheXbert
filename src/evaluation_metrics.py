import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from sklearn.metrics import f1_score, confusion_matrix
from utils import CONDITIONS

def compute_mention_f1(y_true, y_pred):
    """Compute the mention F1 score as in CheXpert paper
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions

    @returns res (list): List of 14 scalars
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == 2] = 1
        y_true[j][y_true[j] == 3] = 1
        y_pred[j][y_pred[j] == 2] = 1
        y_pred[j][y_pred[j] == 3] = 1

    res = []
    for j in range(len(y_true)):
        res.append(f1_score(y_true[j], y_pred[j], pos_label=1))

    return res

def compute_blank_f1(y_true, y_pred):
    """Compute the blank F1 score 
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions
                                                                         
    @returns res (list): List of 14 scalars                           
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == 2] = 1
        y_true[j][y_true[j] == 3] = 1
        y_pred[j][y_pred[j] == 2] = 1
        y_pred[j][y_pred[j] == 3] = 1

    res = []
    for j in range(len(y_true)):
        res.append(f1_score(y_true[j], y_pred[j], pos_label=0))

    return res

def compute_negation_f1(y_true, y_pred):
    """Compute the negation F1 score as in CheXpert paper
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions   

    @returns res (list): List of 14 scalars
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == 3] = 0
        y_true[j][y_true[j] == 1] = 0
        y_pred[j][y_pred[j] == 3] = 0
        y_pred[j][y_pred[j] == 1] = 0

    res = []
    for j in range(len(y_true)-1):
        res.append(f1_score(y_true[j], y_pred[j], pos_label=2))

    res.append(0) #No Finding gets score of zero
    return res

def compute_positive_f1(y_true, y_pred):
    """Compute the positive F1 score
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions 

    @returns res (list): List of 14 scalars
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == 3] = 0
        y_true[j][y_true[j] == 2] = 0
        y_pred[j][y_pred[j] == 3] = 0
        y_pred[j][y_pred[j] == 2] = 0

    res = []
    for j in range(len(y_true)):
        res.append(f1_score(y_true[j], y_pred[j], pos_label=1))

    return res

def compute_uncertain_f1(y_true, y_pred):
    """Compute the negation F1 score as in CheXpert paper
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions

    @returns res (list): List of 14 scalars
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == 2] = 0
        y_true[j][y_true[j] == 1] = 0
        y_pred[j][y_pred[j] == 2] = 0
        y_pred[j][y_pred[j] == 1] = 0

    res = []
    for j in range(len(y_true)-1):
        res.append(f1_score(y_true[j], y_pred[j], pos_label=3))

    res.append(0) #No Finding gets a score of zero
    return res

def get_weighted_f1_weights(train_path_or_csv):
    """Compute weights used to obtain the weighted average of
       mention, negation and uncertain f1 scores. 
    @param train_path_or_csv: A path to the csv file or a dataframe

    @return weight_dict (dictionary): maps conditions to a list of weights, the order
                                      in the lists is negation, uncertain, positive 
    """
    if isinstance(train_path_or_csv, str):
        df = pd.read_csv(train_path_or_csv)
    else:
        df = train_path_or_csv
    df.replace(0, 2, inplace=True)
    df.replace(-1, 3, inplace=True)
    df.fillna(0, inplace=True)

    weight_dict = {}
    for cond in CONDITIONS:
        weights = []
        col = df[cond]

        mask = col == 2
        weights.append(mask.sum())

        mask = col == 3
        weights.append(mask.sum())

        mask = col == 1
        weights.append(mask.sum())

        if np.sum(weights) > 0:
            weights = np.array(weights)/np.sum(weights)
        weight_dict[cond] = weights
    return weight_dict

def weighted_avg(scores, weights):
    """Compute weighted average of scores
    @param scores(List): the task scores
    @param weights (List): corresponding normalized weights

    @return (float): the weighted average of task scores
    """
    return np.sum(np.array(scores) * np.array(weights))
