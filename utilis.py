"""utillis for every implementation.
"""
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
def is_leave(E, F, curr_major, major_father,  M=0):
    if F is None or len(F) == 0:  # affect only at discrete features
        return True, major_father

    if E.shape[0] < M:  # pruning by parameter M
        return True, major_father

    if E[E['diagnosis'] == curr_major].shape[0] == E.shape[0]:  # this is leave
        return True, curr_major

    return False, curr_major
