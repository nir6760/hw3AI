"""class of ID3 algo.
"""
import utilis
import math
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class Tree:

    def __init__(self, f, childs, default_val):
        self.feature = f
        self.children = childs
        self.class_of_node = default_val


class ID3:
    def __init__(self):
        self.tree = None

    # train ID3 , X- data:train, Y- features
    def fit(self, X, Y):
        c = utilis.majority_class(X)
        self.tree = self.TDIDT(X, Y, c, ID3.MaxIG)

    # train ID3 , X- data:train, Y- features with early pruning by parameter M
    def fitEarlyPruning(self, X, Y, M):
        c = utilis.majority_class(X)
        self.tree = self.TDIDT(X, Y, c, ID3.MaxIG, M)

    # test ID3 and return the true_positive rate, X- data:test
    def predict(self, X):
        size_X = X.shape[0]
        if size_X == 0:
            raise Exception('There is no test, you can go home')
        cnt_true_positive = 0
        for index, row in X.iterrows():
            if ID3.classify(row, self.tree) == row['diagnosis']:
                cnt_true_positive += 1
        return cnt_true_positive / size_X

    # classify object
    @staticmethod
    def classify(object, tree):
        if tree is None:
            raise Exception("my friend, you need to train me first")
        if tree.children is None:
            return tree.class_of_node

        attribute = tree.feature[0]
        thres = tree.feature[1]
        # print('attribute ', attribute , ' - threshold ', thres)

        going_to = 0
        if object[attribute] >= thres:  # feature is 0
            going_to = 1

        # child[0] - val of feature, child[1] - subtree
        next_subtree = tree.children[going_to][1]
        return ID3.classify(object, next_subtree)

    # make the discrete functions, for dynamic partition
    @staticmethod
    def make_discrete(f, E):

        best_thres = -1
        best_ig = -1
        relevant_df_np = E[['diagnosis', f]].sort_values(by=[f]).to_numpy()

        curr_thres_func = lambda i: (relevant_df_np[i - 1][1] + relevant_df_np[i][
            1]) / 2  # if we dont have 2 its a leave
        h_func = lambda pb, pm: -pb * math.log(pb, 2) - pm * math.log(pm, 2) if (pb != 0 and pm != 0) else 0  # Entropy

        size_examples = relevant_df_np.shape[0]
        numB_total = np.count_nonzero(relevant_df_np == 'B')
        p_B_total = numB_total / size_examples
        p_M_total = 1 - p_B_total
        h_e = h_func(p_B_total, p_M_total)

        last_val = relevant_df_np[0][1]

        it = 0
        numB_at0 = 0
        while it < len(relevant_df_np) - 1:

            while it < len(relevant_df_np) - 1 and relevant_df_np[it][1] <= last_val:
                numB_at0 += 1 if relevant_df_np[it][0] == 'B' else 0
                it += 1

            last_val = relevant_df_np[it][1]
            curr = curr_thres_func(it)

            size_examples0 = it
            size_examples1 = size_examples - size_examples0

            p_B0 = numB_at0 / size_examples0
            p_M0 = 1 - p_B0

            p_B1 = (numB_total - numB_at0) / size_examples1
            p_M1 = 1 - p_B1

            ig = h_e - ((size_examples0 * h_func(p_B0, p_M0) +
                         size_examples1 * h_func(p_B1, p_M1))
                        / size_examples)

            if ig > best_ig:
                best_thres = curr
                best_ig = ig

        return best_ig, best_thres

    # function which select the best feature for current node in ID3 Tree
    # select with dynamic partition
    @staticmethod
    def MaxIG(F, E):
        best_f = ['Default', -1]
        best_ig = -1
        for f in F:
            curr_f_best_ig, curr_f_best_thres = ID3.make_discrete(f, E)
            if curr_f_best_ig >= best_ig:
                best_f = [f, curr_f_best_thres]
                best_ig = curr_f_best_ig

        return best_f  # argmax

    # create TDIDT Tree from given E and F with select feature function,
    # M is for pruning
    @staticmethod
    def TDIDT(E, F, Default, SelectFeature, M=0):
        if E.empty:  # there is no examples
            return Tree(None, None, Default)
        c = utilis.majority_class(E)
        if utilis.is_leave(E, F, c, M):
            return Tree(None, None, c)
        f = SelectFeature(F, E)

        # F.remove(f[0])
        # F = delete_from_features(F, f) # on discrete cases

        subexamples0 = E[E[f[0]] < f[1]]
        subexamples1 = E[E[f[0]] >= f[1]]
        child0 = [0, ID3.TDIDT(subexamples0, F, c, SelectFeature, M)]
        child1 = [1, ID3.TDIDT(subexamples1, F, c, SelectFeature, M)]
        subtrees = [child0, child1]
        return Tree(f, subtrees, c)

def experiment():
    E_train, F = utilis.createDF_train()
    M_lst = [1, 2, 3, 5, 8, 16, 30, 50, 80, 120]
    it = 0
    avg_lst = [0 for i in range(len(M_lst))]

    kf = KFold(n_splits=5, shuffle=True, random_state=123456789)  # todo: replace to 205467780
    n_spilit = kf.get_n_splits()
    for train_index, test_index in kf.split(E_train):
        for it in range(len(M_lst)):
            id3 = ID3()
            id3.fitEarlyPruning(E_train.loc[train_index], F, M_lst[it])
            avg_lst[it] += id3.predict(E_train.loc[test_index])/n_spilit
    print(avg_lst)

    plt.plot(M_lst, avg_lst)

    # naming the x axis
    plt.xlabel('M values - axis')
    # naming the y axis
    plt.ylabel('avarge predication - axis')
    plt.title('Section 3.3')
    plt.show()






if __name__ == '__main__':
    experiment()

    #E_train, F = utilis.createDF_train()
    #E_test, F = utilis.createDF_test()
    #id3 = ID3()
    #id3.fit(E_train, F)
    #print(id3.predict(E_test))
