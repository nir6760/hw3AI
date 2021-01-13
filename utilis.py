"""utillis for every implementation.
"""
import numpy as np
import pandas as pd

path_train = 'train.csv'
path_test = 'test.csv'


# return examples (as dataframe) and features(as list) from training set
def createDF_train():
    df = pd.read_csv(path_train)
    features = df.columns.to_list()
    features.remove('diagnosis')
    return df, features


# return examples (as dataframe) and features(as list) from test set
def createDF_test():
    df = pd.read_csv(path_test)
    features = df.columns.to_list()
    features.remove('diagnosis')
    return df, features


# find the majority class and return it, E is data frame
def majority_class(E):
    count_B = E[E['diagnosis'] == 'B'].shape[0]
    count_M = E[E['diagnosis'] == 'M'].shape[0]
    win = 'B'
    if count_M > count_B:
        win = 'M'
    return win


# find if there are no examples, or they all the same of if we need pruning
# at continues space F won't be None because we don't remove features
def is_leave(E, F, c, M=0):
    if F is None or len(F) == 0:
        return True


    if E.shape[0] <= M:  # pruning by parameter M
        return True
    if E[E['diagnosis'] != c].shape[0] > 0:  # not all values are the same
        return False
    return True
