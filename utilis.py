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
    count_B = 0
    count_M = 0
    df_diagnosis = E['diagnosis']
    for t in df_diagnosis:
        if t == 'M':
            count_M += 1
        if t == 'B':
            count_B += 1
    win = 'B'
    if count_M > count_B:
        win = 'M'
    return win

# find if there are no examples, or they all the same
# in continues space F won't be None because we don't remove features
def is_node(E, F, c):
    if F is None or len(F) == 0:
        return True
    df_diagnosis = E['diagnosis']
    for t in df_diagnosis:
        if t != c:
            return False
    return True
